#!/usr/bin/env python3
"""
RL Training Script for Autonomous CMM Touch-Point Planning ("Next-Best-Touch").
Uses MaskablePPO from sb3_contrib with Invalid Action Masking to train an inspection policy
that maximizes surface uncertainty reduction, guarantees 4-point coverage across all 5 faces,
and minimizes Chebyshev clearance-box travel time.
"""

import os
import sys
import argparse
import time
import numpy as np

from cmm_env import CMMTouchPointEnv
from pointcloud_preprocessor import PointCloudPreprocessor, PreprocessedCandidateData


def train(npz_path: str = 'candidates_256.npz',
          total_timesteps: int = 40000,
          model_save_path: str = 'cmm_maskable_ppo.zip',
          domain_randomize: bool = True):
    """
    Train MaskablePPO agent on the preprocessed CMM candidate environment.
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback

    class CMMMetricsCallback(BaseCallback):
        """Logs episode returns, mean error, travel time, and face coverage."""
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_travel_times = []
            self.episode_uncs = []

        def _on_step(self) -> bool:
            # Check for episode termination
            for env_idx, done in enumerate(self.locals.get('dones', [])):
                if done:
                    info = self.locals['infos'][env_idx]
                    ep_rew = self.locals.get('rewards', [0.0])[env_idx]
                    t_time = info.get('total_travel_time', 0.0)
                    unc = info.get('mean_uncertainty', 0.0)
                    self.episode_travel_times.append(t_time)
                    self.episode_uncs.append(unc)

                    if len(self.episode_travel_times) % 10 == 0:
                        avg_time = np.mean(self.episode_travel_times[-10:])
                        avg_unc = np.mean(self.episode_uncs[-10:])
                        print(
                            f"[TRAIN STEP {self.num_timesteps:6d}] "
                            f"Recent 10 eps -> Avg Travel Time: {avg_time:.2f}s | "
                            f"Avg Remaining Uncertainty: {avg_unc*100:.1f}% | "
                            f"Faces touched: {info.get('face_touch_counts')}"
                        )
            return True

    def mask_fn(env: CMMTouchPointEnv) -> np.ndarray:
        return env.action_masks()

    print(f"[INFO] Initializing CMMTouchPointEnv with '{npz_path}'...")
    base_env = CMMTouchPointEnv(
        npz_path=npz_path,
        K=256,
        max_touches=20,
        domain_randomize=domain_randomize
    )
    env = ActionMasker(base_env, mask_fn)

    print("[INFO] Initializing MaskablePPO (MlpPolicy, n_steps=2048, batch_size=64, lr=3e-4, gamma=0.99)...")
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    print(f"[INFO] Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    callback = CMMMetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    duration = time.time() - start_time
    print(f"[SUCCESS] Training completed in {duration:.1f} seconds ({total_timesteps / duration:.1f} steps/s)!")

    # Save model
    model.save(model_save_path)
    workspace_save = os.path.join('/home/raghav/CMM_Automation', os.path.basename(model_save_path))
    if os.path.abspath(workspace_save) != os.path.abspath(model_save_path):
        model.save(workspace_save)
    print(f"[SUCCESS] Trained policy saved to: {model_save_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO for CMM Next-Best-Touch Planning")
    parser.add_argument("--npz", type=str, default="candidates_256.npz", help="Path to preprocessed candidates NPZ")
    parser.add_argument("--timesteps", type=int, default=40000, help="Total training timesteps (default: 40000)")
    parser.add_argument("--output", type=str, default="cmm_maskable_ppo.zip", help="Model output file")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable domain randomization")
    args = parser.parse_args()

    train(
        npz_path=args.npz,
        total_timesteps=args.timesteps,
        model_save_path=args.output,
        domain_randomize=not args.no_domain_rand
    )


if __name__ == "__main__":
    main()
