import sys
import os
import random
import torch
import numpy as np
from typing import Dict
import logging
import asyncio
import traceback
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def set_seed(seed: int = 42):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Logging setup: rich handler, silence noisy modules ---
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)]
)
logger = logging.getLogger(__name__)

# Silence repetitive sub-module logs (data loading, memory ops, env init, config debug)
_NOISY_LOGGERS = [
    "market_maker.memory", "market_maker.environment", "market_maker.live_env",
    "core.config", "environment.sim_env", "environment.base_environment",
    "environment.vec_env", "data.loader", "data.data_loader", "data.recorder",
    "data.processor", "data.binance", "data.binance.stream",
    "models.mamba_lob", "models.rl.PPO_agent", "models.rl.memory",
]
for name in _NOISY_LOGGERS:
    logging.getLogger(name).setLevel(logging.WARNING)


def _silence_noisy_loggers():
    """Catch-all: silence any logger that floods during env/data init."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if any(k in name.lower() for k in [
            "data", "loader", "cache", "memory", "config",
            "sim_env", "base_env", "mamba", "ppo", "environment", "vec_env",
            "processor", "recorder", "binance", "stream",
        ]):
            logging.getLogger(name).setLevel(logging.WARNING)


_silence_noisy_loggers()

# Configuration des chemins d'importation
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'environment'))
sys.path.append(os.path.join(current_dir, 'models'))
sys.path.append(os.path.join(current_dir, 'data'))
sys.path.append(os.path.join(current_dir, 'core'))

# Import des modules
from core.config import MarketConfig, RLConfig, ModelConfig, SimulationConfig
from models.rl.PPO_agent import PPOAgent
from environment.sim_env import SimulationMarketMakerEnv
from environment.vec_env import AsyncVectorEnv

class ModelCheckpoint:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_reward = float('-inf')

    def save(self, agent, episode, metrics, is_best=False):
        checkpoint = {
            'episode': episode,
            'agent_state': agent.state_dict(),
            'optimizer_state': agent.optimizer.state_dict(),
            'metrics': metrics,
            'best_reward': self.best_reward
        }

        if episode % 10 == 0:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_{episode}.pt')
            torch.save(checkpoint, path)

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, path)
            self.best_reward = metrics['episode_rewards'][-1]

    def load(self, checkpoint_path, agent):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.load_state_dict(checkpoint['agent_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        return checkpoint

async def train(config: Dict, train_data_path: str, checkpoint_dir: str = "checkpoints", resume_from: str = None):
    logger.info("Starting training...")
    writer = None

    try:
        # Reproducibility
        seed = config['training'].get('seed', 42)
        set_seed(seed)
        logger.info(f"Random seed set to {seed}")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Configurations
        market_config = MarketConfig(
            symbol=config['market']['symbol'],
            tick_size=config['market']['tick_size'],
            min_qty=config['market']['lot_size'],
            min_notional=10.0,
            max_position=config['rl']['max_position'],
            update_interval=0.1
        )
        
        rl_config = RLConfig(
            zeta=config['rl']['zeta'],
            gamma=config['rl']['gamma'],
            max_bias=config['rl']['max_bias'],
            max_spread=config['rl']['max_spread'],
            min_spread=config['rl'].get('min_spread', 0.001),
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            max_steps=config['rl']['max_steps'],
            execution_intensity=config['rl']['execution_intensity'],
            num_epochs=config['rl']['num_epochs'],
            clip_ratio=config['rl']['clip_ratio'],
            target_kl=config['rl']['target_kl'],
            gae_lambda=config['rl']['gae_lambda'],
            max_grad_norm=config['rl']['max_grad_norm'],
            buffer_size=config['training']['buffer_size'],
            # SOTA optimizer settings
            optimizer=config['rl'].get('optimizer', 'muon'),
            weight_decay=config['rl'].get('weight_decay', 0.1)
        )
        # torch.compile optimization (can be disabled via CLI --no-compile)
        rl_config.use_torch_compile = config['rl'].get('use_torch_compile', True)

        sim_config = SimulationConfig(
            maker_fee=config['market'].get('maker_fee', 0.001),
            taker_fee=config['market'].get('taker_fee', 0.001),
            fee_curriculum=config['market'].get('fee_curriculum', False)
        )
        model_config = ModelConfig(**config['model'])
        
        # Agent PPO with integrated LOBModel backbone
        logger.info("Initializing PPO agent with LOBModel backbone...")
        agent = PPOAgent(
            model_config=model_config,
            rl_config=rl_config,
            device=device,
            pretrained_path=config['training'].get('pretrained_path'),
            freeze_backbone=config['training'].get('freeze_backbone', False),
            n_episodes=config['training']['n_episodes']
        )
        
        # Initialisation checkpoints et tensorboard
        checkpoint = ModelCheckpoint(checkpoint_dir)
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        
        # Métriques
        metrics = {
            'episode_rewards': [],
            'train_losses': {
                'actor_loss': [],
                'critic_loss': [],
                'entropy': [],
                'kl_divergence': []
            },
            'inventories': [],
            'pnl_metrics': {
                'realized_pnl': [],
                'unrealized_pnl': [],
                'total_pnl': []
            },
            'execution_metrics': {
                'fill_rates': [],
                'spreads': []
            }
        }
        
        # Reprise d'entraînement si nécessaire
        start_episode = 0
        if resume_from:
            loaded = checkpoint.load(resume_from, agent)
            start_episode = loaded['episode'] + 1
            metrics = loaded['metrics']
            logger.info(f"Resuming training from episode {start_episode}")
        
        # Silence loggers again (imports may have registered new ones)
        _silence_noisy_loggers()

        # Create vectorized train environments for parallel data collection
        n_envs = config['training'].get('n_envs', 8)

        def make_train_env():
            return SimulationMarketMakerEnv(
                rl_config=rl_config,
                market_config=market_config,
                sim_config=sim_config,
                data_path=train_data_path,
                split='train'
            )

        vec_env = AsyncVectorEnv(make_train_env, n_envs=n_envs)
        # Single env fallback for backward compat
        env = make_train_env()

        eval_env = SimulationMarketMakerEnv(
            rl_config=rl_config,
            market_config=market_config,
            sim_config=sim_config,
            data_path=train_data_path,
            split='eval'
        )

        # Eval config
        eval_interval = config['training'].get('eval_interval', 20)
        eval_episodes = config['training'].get('eval_episodes', 3)
        best_eval_reward = float('-inf')

        # Policy collapse detection
        collapse_window = 30  # episodes to look back (reduced for faster detection)
        entropy_history = []
        reward_history_collapse = []
        kl_history = []  # Track KL for immediate collapse detection
        last_good_checkpoint_path = None
        collapse_count = 0
        max_collapses = 5  # max rollbacks before giving up

        # --- Training banner ---
        n_episodes = config['training']['n_episodes']
        console.print(Panel.fit(
            f"[bold cyan]PPO Market Maker Training[/]\n"
            f"Device: [green]{device}[/] | Episodes: [yellow]{n_episodes}[/] | "
            f"Vec Envs: [yellow]{n_envs}[/] | Batch: [yellow]{config['training']['batch_size']}[/] | "
            f"LR: [yellow]{config['training']['learning_rate']}[/]",
            title="[bold]SOTA Pipeline", border_style="blue"
        ))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[status]}"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
        )

        with progress:
            train_task = progress.add_task("Training", total=n_episodes - start_episode, status="starting...")

            for episode in range(start_episode, n_episodes):
                # === FEE CURRICULUM LEARNING ===
                # Update fees based on episode (start low, increase gradually)
                if sim_config.fee_curriculum:
                    old_maker = sim_config.maker_fee
                    maker_fee, taker_fee = sim_config.update_fees(episode)
                    if maker_fee != old_maker:
                        logger.info(f"[Curriculum] Episode {episode}: fees updated to "
                                    f"maker={maker_fee*10000:.1f}bps, taker={taker_fee*10000:.1f}bps")
                        # Sync fees to all vectorized environments
                        vec_env.call_method('set_fees', maker_fee, taker_fee)

                # Run one episode on primary env (for metrics tracking)
                state = await env.reset()
                agent.reset_episode()
                episode_reward = 0
                step = 0
                ep_fills = 0
                ep_spread_sum = 0.0
                ep_spread_count = 0
                ep_max_inv = 0.0  # Track max absolute inventory during episode
                ep_inv_sum = 0.0  # For average inventory

                # Sync learned risk aversion γ to environment (Avellaneda-Stoikov)
                env.set_risk_aversion(agent.get_risk_aversion())

                while True:
                    action, log_prob, value = agent.select_action(state)
                    next_state, reward, done, info = await env.step(action)

                    step_execs = info.get('executions', [])
                    ep_fills += len(step_execs)
                    quotes = info.get('quotes', {})
                    if quotes.get('bid_price', 0) > 0 and quotes.get('ask_price', 0) > 0:
                        ep_spread_sum += quotes['ask_price'] - quotes['bid_price']
                        ep_spread_count += 1

                    agent.track_adverse_selection(info)

                    # Track PnL for variance estimation (Lagrangian learning of γ)
                    ep_metrics = info.get('metrics', {})
                    step_pnl = ep_metrics.get('total_pnl', 0)
                    agent.track_pnl(step_pnl)

                    # Track inventory during episode (not just final)
                    step_inv = abs(ep_metrics.get('inventory', 0))
                    ep_max_inv = max(ep_max_inv, step_inv)
                    ep_inv_sum += step_inv

                    agent.store_transition(
                        state=state, action=action, reward=reward,
                        next_state=next_state, log_prob=log_prob, value=value, done=done
                    )

                    state = next_state
                    episode_reward += reward
                    step += 1

                    if done:
                        break

                # Parallel data collection from vectorized envs
                # Each env's trajectory is collected into per-env buffers, then
                # stored sequentially in the agent's memory. This ensures GAE
                # computes correctly within each trajectory (done boundaries reset
                # the advantage), unlike the previous interleaved storage which
                # mixed transitions from different envs and corrupted GAE.
                if n_envs > 1:
                    # Sync learned risk aversion to all vec envs
                    gamma_risk = agent.get_risk_aversion()
                    vec_env.call_method('set_risk_aversion', gamma_risk)

                    vec_states = await vec_env.reset()
                    # Initialize per-env histories for this rollout
                    agent.init_vec_histories(n_envs)

                    vec_dones = [False] * n_envs
                    # Per-env trajectory buffers (stored separately, flushed sequentially)
                    env_trajectories = [[] for _ in range(n_envs)]

                    while not all(vec_dones):
                        # BATCHED forward pass: single GPU call for all envs (8x speedup)
                        vec_actions, vec_log_probs, vec_values = agent.select_action_batch(
                            vec_states, vec_dones
                        )

                        next_vec_states, vec_rewards, new_dones, vec_infos = await vec_env.step(vec_actions)

                        for i in range(n_envs):
                            if not vec_dones[i]:
                                agent.track_adverse_selection(vec_infos[i])
                                env_trajectories[i].append({
                                    'state': vec_states[i], 'action': vec_actions[i],
                                    'reward': vec_rewards[i], 'next_state': next_vec_states[i],
                                    'log_prob': vec_log_probs[i], 'value': vec_values[i],
                                    'done': new_dones[i]
                                })

                        vec_states = next_vec_states
                        vec_dones = [vec_dones[i] or new_dones[i] for i in range(n_envs)]

                    # Flush per-env trajectories sequentially into agent memory.
                    # Each trajectory is contiguous, so GAE will correctly reset
                    # at each done boundary without cross-env contamination.
                    # OPTIMIZED: Use store_trajectory for 10x faster batch storage
                    for env_traj in env_trajectories:
                        agent.store_trajectory(env_traj)

                # PPO Update
                update_metrics = agent.update()

                if update_metrics:
                    cur_entropy = update_metrics.get('entropy', 0)
                    cur_kl = update_metrics.get('approx_kl', 0)
                    entropy_history.append(cur_entropy)
                    reward_history_collapse.append(episode_reward)
                    kl_history.append(cur_kl)

                    # === IMMEDIATE KL COLLAPSE DETECTION ===
                    # If KL spikes above 1.0, the policy just made a catastrophic update
                    kl_collapse = abs(cur_kl) > 1.0 and collapse_count < max_collapses
                    if kl_collapse:
                        collapse_count += 1
                        logger.warning(f"POLICY COLLAPSE (KL={cur_kl:.4f}) at ep {episode} "
                                       f"(#{collapse_count}/{max_collapses})")
                        writer.add_scalar('Collapse/detected', collapse_count, episode)

                        if last_good_checkpoint_path and os.path.exists(last_good_checkpoint_path):
                            logger.warning(f"Rolling back to {last_good_checkpoint_path}")
                            checkpoint.load(last_good_checkpoint_path, agent)
                            entropy_history.clear()
                            reward_history_collapse.clear()
                            kl_history.clear()

                    # === GRADUAL COLLAPSE DETECTION (entropy/reward) ===
                    if not kl_collapse and len(entropy_history) > collapse_window and collapse_count < max_collapses:
                        recent_ent = np.mean(entropy_history[-collapse_window//2:])
                        older_ent = np.mean(entropy_history[-collapse_window:-collapse_window//2])
                        recent_rew = np.mean(reward_history_collapse[-collapse_window//2:])
                        older_rew = np.mean(reward_history_collapse[-collapse_window:-collapse_window//2])

                        entropy_collapsed = recent_ent < 0.1 * older_ent and older_ent > 0.1
                        reward_collapsed = (older_rew > 0 and recent_rew < 0.2 * older_rew)

                        if entropy_collapsed or reward_collapsed:
                            collapse_count += 1
                            reason = "entropy" if entropy_collapsed else "reward"
                            logger.warning(f"POLICY COLLAPSE ({reason}) at ep {episode} (#{collapse_count}/{max_collapses})")
                            writer.add_scalar('Collapse/detected', collapse_count, episode)

                            if last_good_checkpoint_path and os.path.exists(last_good_checkpoint_path):
                                logger.warning(f"Rolling back to {last_good_checkpoint_path}")
                                checkpoint.load(last_good_checkpoint_path, agent)
                                entropy_history.clear()
                                reward_history_collapse.clear()
                                kl_history.clear()

                    # Save good checkpoint for rollback (every 10 episodes, much more frequent)
                    if episode % 10 == 0 and episode > 0 and len(entropy_history) > 5:
                        if np.mean(entropy_history[-5:]) > 0.3 and (len(kl_history) < 3 or np.mean(kl_history[-3:]) < 0.1):
                            last_good_checkpoint_path = os.path.join(checkpoint_dir, 'last_good.pt')
                            torch.save({
                                'episode': episode, 'agent_state': agent.state_dict(),
                                'optimizer_state': agent.optimizer.state_dict(),
                                'metrics': metrics, 'best_reward': checkpoint.best_reward
                            }, last_good_checkpoint_path)

                    metrics['train_losses']['actor_loss'].append(update_metrics['actor_loss'])
                    metrics['train_losses']['critic_loss'].append(update_metrics['critic_loss'])
                    metrics['train_losses']['entropy'].append(update_metrics['entropy'])
                    metrics['train_losses']['kl_divergence'].append(update_metrics.get('approx_kl', 0))
                    writer.add_scalar('Loss/actor', update_metrics['actor_loss'], episode)
                    writer.add_scalar('Loss/critic', update_metrics['critic_loss'], episode)
                    writer.add_scalar('Loss/entropy', update_metrics['entropy'], episode)
                    writer.add_scalar('Loss/approx_kl', update_metrics.get('approx_kl', 0), episode)
                    # Adaptive KL metrics
                    writer.add_scalar('AdaptiveKL/target', update_metrics.get('kl_target', 0), episode)
                    writer.add_scalar('AdaptiveKL/beta', update_metrics.get('kl_beta', 0), episode)
                    writer.add_scalar('AdaptiveKL/calibrated', update_metrics.get('kl_calibrated', 0), episode)
                    # Gradient Noise Scale metrics (McCandlish 2018)
                    writer.add_scalar('AdaptiveLR/lr_optimal', update_metrics.get('lr_optimal', 0), episode)
                    writer.add_scalar('AdaptiveLR/b_noise', update_metrics.get('b_noise', 0), episode)
                    writer.add_scalar('AdaptiveLR/grad_snr', update_metrics.get('grad_snr', 0), episode)
                    writer.add_scalar('AdaptiveLR/calibrated', update_metrics.get('lr_calibrated', 0), episode)

                # Episode metrics
                ep_metrics = info.get('metrics', {})
                metrics['episode_rewards'].append(episode_reward)
                metrics['inventories'].append(ep_metrics.get('inventory', 0))
                metrics['pnl_metrics']['total_pnl'].append(ep_metrics.get('total_pnl', 0))
                metrics['pnl_metrics']['realized_pnl'].append(ep_metrics.get('realized_pnl', 0))
                metrics['pnl_metrics']['unrealized_pnl'].append(ep_metrics.get('unrealized_pnl', 0))
                metrics['execution_metrics']['fill_rates'].append(ep_metrics.get('fill_rate', 0))
                metrics['execution_metrics']['spreads'].append(ep_metrics.get('average_spread', 0))

                # Progress bar update
                pnl = ep_metrics.get('total_pnl', 0)
                inv_final = ep_metrics.get('inventory', 0)
                inv_avg = ep_inv_sum / max(1, step)  # Average |inventory| during episode
                ent = update_metrics.get('entropy', 0) if update_metrics else 0
                kl = update_metrics.get('approx_kl', 0) if update_metrics else 0
                a_loss = update_metrics.get('actor_loss', 0) if update_metrics else 0
                lr_current = agent.optimizer.param_groups[0]['lr']

                avg_reward = np.mean(metrics['episode_rewards'][-100:]) if episode >= 100 else episode_reward
                status_str = (
                    f"R={episode_reward:+.1f} avg100={avg_reward:+.1f} "
                    f"PnL={pnl:+.3f} inv={ep_max_inv:.3f}/{inv_avg:.3f} "
                    f"H={ent:.2f} KL={kl:.3f} lr={lr_current:.1e}"
                )
                progress.update(train_task, advance=1, status=status_str)

                # Detailed log every 10 episodes
                if episode % 10 == 0:
                    logger.info(
                        f"[bold]Ep {episode}[/] | reward={episode_reward:+.2f} | "
                        f"PnL={pnl:+.4f} | inv_max={ep_max_inv:.3f} inv_avg={inv_avg:.3f} | fills={ep_fills} | "
                        f"actor={a_loss:.4f} | entropy={ent:.3f} | KL={kl:.4f} | "
                        f"lr={lr_current:.2e}"
                    )

                if episode >= 100:
                    writer.add_scalar('Reward/avg_last_100_episodes', avg_reward, episode)

                # TensorBoard
                writer.add_scalar('Reward/episode', episode_reward, episode)
                writer.add_scalar('PnL/total', ep_metrics.get('total_pnl', 0), episode)
                writer.add_scalar('PnL/realized', ep_metrics.get('realized_pnl', 0), episode)
                writer.add_scalar('PnL/unrealized', ep_metrics.get('unrealized_pnl', 0), episode)
                writer.add_scalar('Inventory/episode', ep_metrics.get('inventory', 0), episode)

                avg_spread = ep_spread_sum / max(1, ep_spread_count)
                fill_rate = ep_fills / max(1, step)
                pnl_per_fill = ep_metrics.get('total_pnl', 0) / max(1, ep_fills)
                writer.add_scalar('MM/spread_moyen', avg_spread, episode)
                writer.add_scalar('MM/fill_rate', fill_rate, episode)
                writer.add_scalar('MM/pnl_par_fill', pnl_per_fill, episode)
                writer.add_scalar('MM/fills_total', ep_fills, episode)
                writer.add_scalar('MM/inventory_abs', abs(ep_metrics.get('inventory', 0)), episode)

                writer.add_scalar('Lagrangian/lambda_risk', agent.lagrangian_risk.item(), episode)
                writer.add_scalar('Lagrangian/lambda_entropy', agent.lagrangian_ent.item(), episode)
                writer.add_scalar('Lagrangian/lambda_as', agent.lagrangian_as.item(), episode)
                toxic_ratio = agent._toxic_fills / max(1, agent._total_fills)
                writer.add_scalar('Lagrangian/toxic_fill_ratio', toxic_ratio, episode)

                # === THEORETICALLY GROUNDED PARAMETERS ===
                # Avellaneda-Stoikov risk aversion (learned via Lagrangian dual)
                writer.add_scalar('Theory/risk_aversion_gamma', agent.get_risk_aversion(), episode)
                # Adaptive KL penalty coefficient (Schulman 2017)
                writer.add_scalar('Theory/kl_beta', agent.kl_beta, episode)
                # SAC automatic entropy temperature (Haarnoja 2018)
                if agent.auto_entropy_tuning:
                    writer.add_scalar('Theory/alpha', agent.log_alpha.exp().item(), episode)
                # PnL variance (for risk constraint)
                if len(agent._pnl_history) >= 100:
                    import torch as _t
                    pnl_var = _t.tensor(list(agent._pnl_history)).var().item()
                    writer.add_scalar('Theory/pnl_variance', pnl_var, episode)

                # === FEE CURRICULUM ===
                writer.add_scalar('Curriculum/maker_fee_bps', sim_config.maker_fee * 10000, episode)
                writer.add_scalar('Curriculum/taker_fee_bps', sim_config.taker_fee * 10000, episode)
                writer.add_scalar('Curriculum/round_trip_bps', (sim_config.maker_fee + sim_config.taker_fee) * 10000, episode)

                if hasattr(agent.heads, 'actor_logstd_net'):
                    logstd_w_norm = sum(p.data.norm().item() for p in agent.heads.actor_logstd_net.parameters())
                    writer.add_scalar('Agent/logstd_net_wnorm', logstd_w_norm, episode)

                # === Evaluation ===
                if (episode + 1) % eval_interval == 0:
                    agent.backbone.eval()
                    agent.heads.eval()
                    eval_rewards = []
                    eval_pnls = []
                    eval_inventories = []

                    for eval_ep in range(eval_episodes):
                        eval_state = await eval_env.reset()
                        agent.reset_episode()
                        eval_reward = 0

                        while True:
                            with torch.no_grad():
                                eval_action, _, _ = agent.select_action_deterministic(eval_state)
                            eval_next_state, eval_r, eval_done, eval_info = await eval_env.step(eval_action)
                            eval_state = eval_next_state
                            eval_reward += eval_r
                            if eval_done:
                                break

                        eval_ep_metrics = eval_info.get('metrics', {})
                        eval_rewards.append(eval_reward)
                        eval_pnls.append(eval_ep_metrics.get('total_pnl', 0))
                        eval_inventories.append(abs(eval_ep_metrics.get('inventory', 0)))

                    avg_eval_reward = np.mean(eval_rewards)
                    avg_eval_pnl = np.mean(eval_pnls)
                    avg_eval_inv = np.mean(eval_inventories)
                    writer.add_scalar('Eval/reward', avg_eval_reward, episode)
                    writer.add_scalar('Eval/pnl', avg_eval_pnl, episode)
                    writer.add_scalar('Eval/inventory_abs', avg_eval_inv, episode)

                    is_new_best = avg_eval_reward > best_eval_reward
                    eval_table = Table(title=f"Eval @ Episode {episode}", border_style="cyan", show_lines=False)
                    eval_table.add_column("Metric", style="bold")
                    eval_table.add_column("Value", justify="right")
                    eval_table.add_row("Reward", f"{avg_eval_reward:+.2f}")
                    eval_table.add_row("PnL", f"{avg_eval_pnl:+.4f}")
                    eval_table.add_row("|Inventory|", f"{avg_eval_inv:.4f}")
                    eval_table.add_row("Best?", "[bold green]NEW BEST[/]" if is_new_best else f"best={best_eval_reward:+.2f}")
                    console.print(eval_table)

                    if is_new_best:
                        best_eval_reward = avg_eval_reward
                        checkpoint.best_reward = best_eval_reward
                        checkpoint.save(agent, episode, metrics, is_best=True)

                # Save metrics + periodic checkpoint
                with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f)
                checkpoint.save(agent, episode, metrics, is_best=False)

        console.print(Panel.fit(
            f"[bold green]Training completed[/] | "
            f"Episodes: {n_episodes} | Best eval reward: {best_eval_reward:+.2f}",
            title="Done", border_style="green"
        ))
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        if writer:
            writer.close()

if __name__ == "__main__":
    import argparse
    from core.auto_config import create_training_config

    parser = argparse.ArgumentParser(
        description="Train Market Maker RL Agent - 100% Data-Driven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FULLY DATA-DRIVEN: Just run 'python train.py' - everything is auto-detected!

From Market Data:
  - gamma, zeta, max_bias, execution_intensity, spread_range
  - tick_size, lot_size, max_position, n_episodes

Derived from Training Params:
  - buffer_size, batch_size, learning_rate, warmup, num_epochs
  - target_kl, clip_ratio

Adaptive (Learned During Training):
  - learning_rate (GradientNoiseScale), clip_ratio (AdaptiveClipRatio)
  - kl_target (AdaptiveKLController), entropy_target (AdaptiveCoverage)
  - risk_aversion (Lagrangian dual, Avellaneda-Stoikov)
        """
    )
    parser.add_argument("--data", type=str, default="data/raw_methusdt",
                        help="Path to data directory (default: data/raw_methusdt)")
    parser.add_argument("--position-usd", type=float, default=1000.0,
                        help="Max position size in USD (default: 1000)")
    parser.add_argument("--maker-fee", type=float, default=None,
                        help="Maker fee (default: 0.075%% Binance+BNB)")
    parser.add_argument("--taker-fee", type=float, default=None,
                        help="Taker fee (default: 0.075%% Binance+BNB)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (default: auto from data quantity)")
    parser.add_argument("--data-reuse", type=float, default=3.0,
                        help="How many times to see each data point (default: 3.0)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint directory (default: checkpoints_{symbol})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable fee curriculum (start with low fees, increase gradually)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (use if you have issues)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel envs (default: auto from GPU memory)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: auto from GPU memory)")
    args = parser.parse_args()

    # Use Binance defaults if not specified
    maker_fee = args.maker_fee if args.maker_fee is not None else 0.00075
    taker_fee = args.taker_fee if args.taker_fee is not None else 0.00075

    # Auto-detect config from data
    data_path = os.path.join(current_dir, args.data) if not os.path.isabs(args.data) else args.data

    print("=" * 60)
    print("100% DATA-DRIVEN CONFIG")
    print("=" * 60)

    config = create_training_config(
        data_path=data_path,
        target_position_usd=args.position_usd,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        n_episodes=args.episodes,  # None = auto-derived from data
        data_reuse_factor=args.data_reuse
    )

    # CLI overrides for GPU scaling
    if args.n_envs is not None:
        config['training']['n_envs'] = args.n_envs
        # Recalculate buffer_size
        config['training']['buffer_size'] = args.n_envs * config['rl']['max_steps']
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    derived = config.get('_derived_params', {})
    n_episodes = config['training']['n_episodes']

    # Show config summary
    print("\n[From Market Data]")
    print(f"  symbol:        {config['market']['symbol']}")
    print(f"  mid_price:     ${config['_auto_detected']['mid_price']:,.4f}")
    print(f"  spread:        {config['_auto_detected']['spread_bps']:.2f} bps")
    print(f"  gamma:         {config['rl']['gamma']:.6f}")
    print(f"  zeta:          {config['rl']['zeta']:.6f}")
    print(f"  exec_intensity:{config['rl']['execution_intensity']:.2f}")

    print("\n[From Data Quantity]")
    total_snapshots = derived.get('total_snapshots')
    if total_snapshots:
        print(f"  total_data:    {total_snapshots:,} snapshots")
    print(f"  n_episodes:    {n_episodes} ({derived.get('n_episodes_source', 'N/A')})")
    print(f"  warmup:        {config['training']['warmup_episodes']} ({derived.get('warmup_formula', '')})")

    print("\n[GPU-Optimized Hyperparameters]")
    print(f"  n_envs:        {config['training']['n_envs']} (parallel environments)")
    print(f"  buffer_size:   {config['training']['buffer_size']}")
    print(f"  batch_size:    {config['training']['batch_size']}")
    print(f"  learning_rate: {config['training']['learning_rate']:.2e}")
    print(f"  num_epochs:    {config['rl']['num_epochs']}")
    print(f"  target_kl:     {config['rl']['target_kl']:.4f}")
    print(f"  clip_ratio:    {config['rl']['clip_ratio']:.3f}")

    print("\n[User Overrides]")
    print(f"  position_usd:  ${args.position_usd:,.0f}")
    print(f"  maker_fee:     {maker_fee*10000:.2f} bps {'(default)' if args.maker_fee is None else '(custom)'}")
    print(f"  taker_fee:     {taker_fee*10000:.2f} bps {'(default)' if args.taker_fee is None else '(custom)'}")

    # Curriculum learning disabled by default (use real fees)
    config['market']['fee_curriculum'] = args.curriculum
    if args.curriculum:
        print(f"\n[Curriculum] ENABLED - starting with low fees")
    else:
        print(f"\n[Fees] Using real fees: {maker_fee*10000:.2f}/{taker_fee*10000:.2f} bps")

    # torch.compile optimization (can be disabled with --no-compile)
    config['rl']['use_torch_compile'] = not args.no_compile
    if args.no_compile:
        print(f"\n[torch.compile] DISABLED")
    else:
        print(f"\n[torch.compile] ENABLED (20-40% speedup)")

    symbol = config['market']['symbol']
    checkpoint_dir = args.checkpoint_dir or f"checkpoints_{symbol.lower()}"

    print(f"\nStarting training for {symbol}...")
    print(f"Checkpoints: {checkpoint_dir}")
    print()

    asyncio.run(train(
        config=config,
        train_data_path=data_path,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume
    ))
