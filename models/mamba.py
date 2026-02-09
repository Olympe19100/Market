import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class MambaBlock(nn.Module):
    """
    A pure PyTorch implementation of Mamba block.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = nn.SiLU()

        # x_proj takes in `x` and outputs (dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        
        # dt_proj projects dt from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.log_A_real = nn.Parameter(torch.log(A)) # (d_inner, d_state)
        
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        x_and_res = self.in_proj(x)  # (B, L, 2*ED)
        (x_val, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # Conv
        x_val = x_val.transpose(1, 2)
        x_val = self.conv1d(x_val)[:, :, :seq_len]
        x_val = x_val.transpose(1, 2)
        x_val = self.activation(x_val)

        # SSM
        # Calculate dt, B, C
        x_dbl = self.x_proj(x_val) # (B, L, dt_rank + 2*d_state)
        (dt, B, C) = x_dbl.split(
            split_size=[self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        # Parameterize dt (softplus)
        dt = F.softplus(dt)

        # Discretize A
        A = -torch.exp(self.log_A_real) # (d_inner, d_state)
        # dA = torch.exp(torch.einsum('b l d, d n -> b l d n', dt, A)) # This is heavy memory-wise
        
        # Sequential scan (slow but correct for pure torch)
        # y_t = A * y_{t-1} + B * x_t ? 
        # Standard: h_t = (I + dA) * h_{t-1} + dB * x_t
        # Real discrete: 
        # A_bar = exp(delta * A)
        # B_bar = (delta * A)^-1 * (exp(delta * A) - I) * delta * B  ~ delta * B
        
        # For efficiency in pure python, we can just do the loop for now. 
        # Ideally we would use the parallel associative scan.
        
        y = self.selective_scan_seq(x_val, dt, A, B, C, self.D)
        
        y = y * self.activation(res)
        output = self.out_proj(y)
        return output

    def selective_scan_seq(self, u, dt, A, B, C, D):
        # u: (B, L, D_inner)
        # dt: (B, L, D_inner)
        # A: (D_inner, D_state)
        # B: (B, L, D_state)
        # C: (B, L, D_state)
        # D: (D_inner)
        
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []
        
        # Discretize A and B dependent on dt
        # Note: In loop, dt changes every step.
        
        # A is diagonal in this impl (S4D), so element-wise mult
        
        for t in range(seq_len):
            dt_t = dt[:, t, :] # (B, D_inner)
            u_t = u[:, t, :] # (B, D_inner)
            B_t = B[:, t, :] # (B, D_state)
            C_t = C[:, t, :] # (B, D_state)
            
            # Discretize A: exp(dt * A)
            # A is (D_inner, D_state). dt is (B, D_inner) -> broadcast?
            # Normally A is (D_inner) or (D_inner, N). Here (D_inner, N).
            # dt is (B, D_inner). Product should be (B, D_inner, N).
            
            dt_A = torch.einsum('bd, dn -> bdn', dt_t, A)
            dA = torch.exp(dt_A) # (B, D_inner, N)
            
            # Discretize B: dt * B
            # B_t is (B, N). dt_t is (B, D_inner).
            # We want (B, D_inner, N).
            dB = torch.einsum('bd, bn -> bdn', dt_t, B_t)
            
            # Update h: h = dA * h + dB * u
            # h is (B, D_inner, N)
            # u_t is (B, D_inner). we need (B, D_inner, N)
            # u broadcast against N.
            
            h = dA * h + dB * u_t.unsqueeze(-1)
            
            # y = C * h + D * u
            # C_t is (B, N). h is (B, D_inner, N).
            # We want y to be (B, D_inner).
            # Sum over N.
            
            y_t = torch.einsum('bn, bdn -> bd', C_t, h)
            y_t = y_t + D * u_t
            
            ys.append(y_t)
            
        return torch.stack(ys, dim=1)

import math

class Mamba(nn.Module):
    def __init__(self, d_model, n_layers, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x # Residual connection commonly used
        return x
