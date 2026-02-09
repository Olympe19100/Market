# environment/vec_env.py
# Asynchronous vectorized environment wrapper for parallel data collection.
# Runs N independent SimulationMarketMakerEnv instances concurrently.

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class AsyncVectorEnv:
    """
    Vectorized async environment: runs N copies of an async env in parallel.
    Each env has its own data split / random offset for decorrelation.

    Usage:
        vec_env = AsyncVectorEnv(env_fn, n_envs=8)
        states = await vec_env.reset()       # list of N state dicts
        states, rewards, dones, infos = await vec_env.step(actions)  # actions: list of N arrays
    """

    def __init__(self, env_fn, n_envs: int = 8):
        """
        Args:
            env_fn: callable() -> env instance (async env with reset/step)
            n_envs: number of parallel environments
        """
        self.n_envs = n_envs

        # Auto-detect optimal number of workers based on CPU cores
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        # Use up to half the cores, capped at n_envs and max 32
        n_workers = min(n_envs, max(4, cpu_cores // 2), 32)

        logger.info(f"Creating {n_envs} environments with {n_workers} workers (CPU: {cpu_cores} cores)...")

        # Parallel environment creation using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            self.envs = list(executor.map(lambda _: env_fn(), range(n_envs)))

        logger.info(f"AsyncVectorEnv initialized with {n_envs} environments")

    async def reset(self) -> List[Dict]:
        """Reset all environments in parallel."""
        tasks = [env.reset() for env in self.envs]
        states = await asyncio.gather(*tasks)
        return list(states)

    async def step(self, actions: List[np.ndarray]) -> Tuple[List[Dict], List[float], List[bool], List[Dict]]:
        """
        Step all environments in parallel.

        Args:
            actions: list of N action arrays

        Returns:
            states: list of N state dicts
            rewards: list of N floats
            dones: list of N bools
            infos: list of N info dicts
        """
        assert len(actions) == self.n_envs, f"Expected {self.n_envs} actions, got {len(actions)}"

        tasks = [env.step(action) for env, action in zip(self.envs, actions)]
        results = await asyncio.gather(*tasks)

        states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]

        return states, rewards, dones, infos

    async def reset_done(self, dones: List[bool]) -> List[Optional[Dict]]:
        """Reset only environments that are done. Returns new states (None for non-done)."""
        tasks = []
        for i, done in enumerate(dones):
            if done:
                tasks.append((i, self.envs[i].reset()))
            else:
                tasks.append((i, None))

        new_states = [None] * self.n_envs
        reset_tasks = [(i, t) for i, t in tasks if t is not None]
        if reset_tasks:
            indices, coros = zip(*reset_tasks)
            results = await asyncio.gather(*coros)
            for idx, state in zip(indices, results):
                new_states[idx] = state

        return new_states

    def call_method(self, method_name: str, *args, **kwargs):
        """Call a method on all environments synchronously.

        Useful for syncing parameters like risk_aversion (gamma) from the agent
        to all environments before data collection.
        """
        for env in self.envs:
            method = getattr(env, method_name, None)
            if method is not None:
                method(*args, **kwargs)
