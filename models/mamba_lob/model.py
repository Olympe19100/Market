# models/mamba_cr/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from typing import Optional, Tuple, List
from core.config import ModelConfig
import logging
from models.mamba import Mamba

# Configuration du logger
logger = logging.getLogger(__name__)

class LOBModel(nn.Module):
    """
    Mamba-LOB: State Space Model for Limit Order Book (LOB) data.
    Replaces the Transformer Attention mechanism with Mamba (SSM) for linear-time sequence modeling.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Store config for use in other methods
        self.config = config
        
        # Extraire les dimensions de la config
        self.input_dim = config.n_features
        self.embed_dim = config.embedding_dim
        # num_heads unused in Mamba but kept in config
        self.sequence_length = config.window_size
        
        # Définir le device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Couches du modèle
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Mamba Backbone
        self.mamba = Mamba(
            d_model=self.embed_dim,
            n_layers=config.num_layers,
            d_state=getattr(config, 'mamba_d_state', 16),
            d_conv=getattr(config, 'mamba_d_conv', 4),
            expand=getattr(config, 'mamba_expand', 2)
        )
        
        # SOTA: Lagrangian Temporal Masking
        # Learnable parameter for dynamic window size optimization
        self.window_cutoff = nn.Parameter(torch.tensor(float(self.sequence_length)))
        self.temperature = getattr(config, 'lagrangian_temperature', 2.0)
        
        self.conv_layers = nn.ModuleList([
            ResidualBlock(self.embed_dim, 256),
            ResidualBlock(256, 384),
            ResidualBlock(384, 512)
        ])
        
        # Encodeur Auxiliaire (Multimodal Fusion)
        self.aux_input_dim = getattr(config, 'n_aux_features', 0)
        self.aux_embed_dim = 64 if self.aux_input_dim > 0 else 0
        
        if self.aux_input_dim > 0:
            self.aux_encoder = nn.Sequential(
                nn.Linear(self.aux_input_dim, self.aux_embed_dim),
                nn.LayerNorm(self.aux_embed_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        
        # Dimension fusionnée: 512 (LOB) + 64 (Aux) = 576
        self.fusion_dim = 512 + self.aux_embed_dim
        
        # Têtes de sortie pour l'action et la value
        self.action_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 2)  # 2 dimensions pour l'action (bias et spread)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )
        
        # Tête de prédiction pour le Self-Supervised Learning (SSL)
        # Reconstruit le vecteur 40-dim à partir des features latentes
        # Tête de prédiction pour le Self-Supervised Learning (SSL) - Probabilistic
        # Output: Mean (40) + Log-Variance (40) = 80 features
        self.forecast_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, self.input_dim * 2) 
        )
        
        # Optimiseur
        self.optimizer = optim.Adam(self.parameters(), lr=getattr(config, 'learning_rate', 3e-4))
        
        # Déplacer le modèle sur le device approprié
        self.to(self.device)
        
        # Initialisation des poids
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, aux_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, features)
        x = self.input_norm(x)
        x = self.embedding(x)
        
        # Mamba Forward Pass (B, L, D) -> (B, L, D)
        x = self.mamba(x)
        
        # Convolution layers
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # Global pooling
        lob_embedding = x.mean(1)  # (batch, features=512)
        
        # Fusion Multimodale
        if self.aux_input_dim > 0 and aux_features is not None:
            aux_embedding = self.aux_encoder(aux_features) # (B, 64)
            # Concaténation
            fused_features = torch.cat([lob_embedding, aux_embedding], dim=1) # (B, 576)
        else:
            # Fallback si pas de features aux ou non configuré
            if self.aux_input_dim > 0:
                # Si le modèle attend des features mais qu'on ne les donne pas (ex: pretrain)
                # On pad avec des zéros
                batch_size = x.shape[0]
                zeros = torch.zeros(batch_size, self.aux_embed_dim, device=x.device)
                fused_features = torch.cat([lob_embedding, zeros], dim=1)
            else:
                fused_features = lob_embedding
        
        # Action et value sur la feature fusionnée
        action_logits = self.action_head(fused_features)
        value = self.value_head(fused_features)
        
        return action_logits, value
    
    def _apply_lagrangian_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft temporal mask based on learnable window_cutoff."""
        b, l, _ = x.shape
        device = x.device
        
        # t = time from "now" (0) to "past" (L-1)
        t = torch.arange(l, device=device).flip(0).unsqueeze(0).expand(b, -1).float()
        
        # Soft mask: sigmoid((cutoff - t) / temperature)
        # Positions near present (t < cutoff) get weight ~1
        # Positions in past (t > cutoff) get weight ~0
        mask = torch.sigmoid((self.window_cutoff - t) / self.temperature)
        
        return x * mask.unsqueeze(-1)
    
    def forward_ssl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass pour le pré-entraînement (SSL) avec masking Lagrangien."""
        # x shape: (batch_size, seq_len, features)
        x = self.input_norm(x)
        
        # Apply Lagrangian temporal masking BEFORE embedding
        # This allows gradient to flow to window_cutoff
        x = self._apply_lagrangian_mask(x)
        
        x = self.embedding(x)
        
        # Mamba
        x = self.mamba(x)  # (B, L, D)
        
        # CNN Layers
        x = x.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.permute(0, 2, 1)
        
        # Global Pooling
        features = x.mean(1)  # (B, D)
        
        # Forecast Next Step
        reconstruction = self.forecast_head(features)  # (B, 80) = mean + log_var
        return reconstruction
        
    def select_action(self, state: dict) -> np.ndarray:
        """Sélectionne une action basée sur l'état actuel."""
        self.eval()  # Mode évaluation
        with torch.no_grad():
            # Préparation des données
            x = torch.FloatTensor(state['lob_features']).unsqueeze(0).to(self.device)
            
            # Forward pass
            action_logits, _ = self.forward(x)
            
            # Normalisation des actions
            action = torch.tanh(action_logits)  # Pour borner entre -1 et 1

            return action.cpu().numpy()[0]
            
    def update(self, batch: List[tuple]) -> float:
        """Met à jour le modèle avec un batch d'expériences."""
        self.train()  # Mode entraînement
        
        # Décompresser le batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir en tensors
        state_batch = torch.FloatTensor(np.array([s['lob_features'] for s in states])).to(self.device)
        action_batch = torch.FloatTensor(np.array(actions)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([s['lob_features'] for s in next_states])).to(self.device)
        done_batch = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Forward pass
        action_logits, values = self.forward(state_batch)
        _, next_values = self.forward(next_state_batch)
        
        # Calcul des avantages
        with torch.no_grad():
            advantages = reward_batch + (1 - done_batch) * 0.99 * next_values.squeeze() - values.squeeze()

        # Pertes
        value_loss = F.mse_loss(values.squeeze(), reward_batch + (1 - done_batch) * 0.99 * next_values.squeeze().detach())
        action_loss = F.mse_loss(action_logits, action_batch)
        loss = value_loss + action_loss
        
        # Log des pertes
        logger.debug("Loss calculated - Value Loss: %f, Action Loss: %f, Total Loss: %f", value_loss.item(), action_loss.item(), loss.item())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=getattr(self.config, 'grad_clip_norm', 1.0) if hasattr(self, 'config') else 1.0)
        self.optimizer.step()
        return loss.item()
    
    def extract_features(self, lob_features: torch.Tensor) -> torch.Tensor:
        """Extrait les caractéristiques de haut niveau."""
        self.eval()
        with torch.no_grad():
            x = self.input_norm(lob_features)
            x = self.embedding(x)
            x = self.mamba(x)

            # Convolution pour l'extraction de features
            x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            x = x.permute(0, 2, 1)  # (batch, seq_len, features)
            
            # Pooling global pour obtenir les caractéristiques finales
            lob_features = x.mean(1)  # (batch, features)
        
        return lob_features

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = F.gelu(x)
        return x
