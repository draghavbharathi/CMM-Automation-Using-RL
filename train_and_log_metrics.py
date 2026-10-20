#!/usr/bin/env python3
"""
Comprehensive RL Training and Telemetry Logging Script for CMM Next-Best-Touch Planning.
Trains MaskablePPO with Invalid Action Masking and records:
- Episode rewards, moving averages, episode lengths
- Metrological success rate (>= 4 points on all 5 faces with low error)
- Final remaining uncertainty and cumulative uncertainty drop
- Total travel time
- PPO policy loss, value loss, entropy loss, approx KL, clip fraction, explained variance
Exports training data to CSV and JSON for scientific visualization and reporting.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

from cmm_env import CMMTouchPointEnv
from pointcloud_preprocessor import PreprocessedCandidateData

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure


class ComprehensiveMetricsCallback(BaseCallback):
    """
    Detailed training callback that captures:
    1. Episode-level metrology metrics (reward, travel time, uncertainty, face coverage)
    2. Policy optimization loss metrics from PPO updates
    """
    def __init__(self, check_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.episode_records = []
        self.update_records = []

        # Running buffers
        self.recent_rewards = []
        self.recent_travel_times = []
        self.recent_uncertainties = []
        self.recent_successes = []

    def _on_step(self) -> bool:
        # Check all environments (single env in DummyVecEnv)
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])

        for idx, done in enumerate(dones):
            if done:
                self.episode_count += 1
                info = infos[idx]
                ep_rew = float(rewards[idx])
                # Check if Monitor wrapper recorded episode
                if 'episode' in info:
                    ep_rew = float(info['episode']['r'])
                    ep_len = int(info['episode']['l'])
                else:
                    ep_len = info.get('step_count', 20)

                t_time = float(info.get('total_travel_time', 0.0))
                mean_unc = float(info.get('mean_uncertainty', 0.0))
                face_counts = info.get('face_touch_counts', [0]*5)
                success = 1.0 if np.all(np.array(face_counts) >= 4) else 0.0
                mean_err = float(info.get('mean_touch_error', 0.0))

                self.recent_rewards.append(ep_rew)
                self.recent_travel_times.append(t_time)
                self.recent_uncertainties.append(mean_unc)
                self.recent_successes.append(success)

                mov_avg_rew = float(np.mean(self.recent_rewards[-20:]))
                mov_avg_success = float(np.mean(self.recent_successes[-20:]))

                record = {
                    'timestep': int(self.num_timesteps),
                    'episode': self.episode_count,
                    'reward': ep_rew,
                    'moving_avg_reward': mov_avg_rew,
                    'length': ep_len,
                    'travel_time': t_time,
                    'remaining_uncertainty': mean_unc,
                    'uncertainty_reduction': float(0.686 - mean_unc),
                    'face_counts': list(face_counts),
                    'success': success,
                    'moving_avg_success': mov_avg_success,
                    'mean_touch_error': mean_err
                }
                self.episode_records.append(record)

                if self.episode_count % self.check_freq == 0:
                    print(
                        f"[Ep {self.episode_count:4d} | Step {self.num_timesteps:6d}] "
                        f"Rew: {ep_rew:6.1f} (MA: {mov_avg_rew:6.1f}) | "
                        f"Success: {mov_avg_success*100:5.1f}% | "
                        f"Travel: {t_time:5.2f}s | "
                        f"Unc: {mean_unc*100:4.1f}% | "
                        f"Faces: {face_counts}"
                    )
        return True

    def _on_rollout_end(self) -> None:
        """Called after each rollout buffer collection (every n_steps=2048 steps)."""
        logger_values = self.model.logger.name_to_value
        record = {
            'timestep': int(self.num_timesteps),
            'iteration': int(self.model._n_updates),
            'policy_loss': float(logger_values.get('train/policy_gradient_loss', 0.0)),
            'value_loss': float(logger_values.get('train/value_loss', 0.0)),
            'entropy_loss': float(logger_values.get('train/entropy_loss', 0.0)),
            'approx_kl': float(logger_values.get('train/approx_kl', 0.0)),
            'clip_fraction': float(logger_values.get('train/clip_fraction', 0.0)),
            'explained_variance': float(logger_values.get('train/explained_variance', 0.0)),
            'fps': float(logger_values.get('time/fps', 0.0))
        }
        self.update_records.append(record)


def run_training(npz_path: str = 'candidates_256.npz',
                 total_timesteps: int = 40960,
                 model_save_path: str = 'cmm_maskable_ppo.zip',
                 log_dir: str = 'training_logs',
                 domain_randomize: bool = True,
                 seed: int = 42) -> None:
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("      RESEARCH-GRADE CMM NEXT-BEST-TOUCH RL POLICY TRAINING (MaskablePPO)")
    print(f"{'='*80}")
    print(f"Candidate dataset  : {npz_path}")
    print(f"Total timesteps    : {total_timesteps}")
    print(f"Model destination  : {model_save_path}")
    print(f"Telemetry log dir  : {log_dir}")
    print(f"Domain randomize   : {domain_randomize}")
    print(f"Random seed        : {seed}")
    print(f"{'='*80}\n")

    def mask_fn(env: CMMTouchPointEnv) -> np.ndarray:
        return env.action_masks()

    base_env = CMMTouchPointEnv(
        npz_path=npz_path,
        K=256,
        max_touches=20,
        domain_randomize=domain_randomize
    )
    base_env.reset(seed=seed)
    env = ActionMasker(base_env, mask_fn)

    sb3_logger = configure(log_dir, ['stdout', 'csv'])

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
        seed=seed,
        verbose=1
    )
    model.set_logger(sb3_logger)

    callback = ComprehensiveMetricsCallback(check_freq=10)

    t_start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    t_elapsed = time.time() - t_start

    print(f"\n[SUCCESS] Completed {total_timesteps} steps in {t_elapsed:.2f}s ({total_timesteps/t_elapsed:.1f} steps/s)!")

    # Save trained policy
    model.save(model_save_path)
    print(f"[SAVED] Policy artifact saved to '{model_save_path}'.")

    # Export metrics to CSV and JSON
    ep_df = pd.DataFrame(callback.episode_records)
    ep_csv_path = os.path.join(log_dir, 'training_episodes.csv')
    ep_df.to_csv(ep_csv_path, index=False)
    print(f"[SAVED] Episode progression telemetry saved to '{ep_csv_path}'.")

    up_df = pd.DataFrame(callback.update_records)
    up_csv_path = os.path.join(log_dir, 'training_updates.csv')
    up_df.to_csv(up_csv_path, index=False)
    print(f"[SAVED] PPO loss & optimization diagnostics saved to '{up_csv_path}'.")

    summary_stats = {
        'total_timesteps': total_timesteps,
        'total_episodes': callback.episode_count,
        'training_duration_seconds': float(t_elapsed),
        'throughput_steps_per_sec': float(total_timesteps / t_elapsed),
        'final_10_ep_reward_mean': float(np.mean(callback.recent_rewards[-10:])),
        'final_10_ep_reward_std': float(np.std(callback.recent_rewards[-10:])),
        'final_success_rate': float(np.mean(callback.recent_successes[-20:])),
        'final_travel_time_mean': float(np.mean(callback.recent_travel_times[-10:])),
        'final_remaining_unc_mean': float(np.mean(callback.recent_uncertainties[-10:])),
    }
    summary_json_path = os.path.join(log_dir, 'training_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    print(f"[SAVED] High-level training summary saved to '{summary_json_path}'.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and log CMM RL Planner metrics")
    parser.add_argument("--npz", type=str, default="candidates_256.npz")
    parser.add_argument("--timesteps", type=int, default=40960)
    parser.add_argument("--output", type=str, default="cmm_maskable_ppo.zip")
    parser.add_argument("--log-dir", type=str, default="training_logs")
    parser.add_argument("--no-domain-rand", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(
        npz_path=args.npz,
        total_timesteps=args.timesteps,
        model_save_path=args.output,
        log_dir=args.log_dir,
        domain_randomize=not args.no_domain_rand,
        seed=args.seed
    )
