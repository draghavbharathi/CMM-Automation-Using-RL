#!/usr/bin/env python3
"""
Academic Benchmarking & Comparative Evaluation Suite for CMM Next-Best-Touch RL.
Evaluates MaskablePPO against 4 metrology and planning baselines:
  1. MaskablePPO (Trained RL Agent)
  2. Greedy Uncertainty Minimizer (Maximizes one-step information gain)
  3. Nearest Neighbor (Minimizes Chebyshev travel time greedily)
  4. Random Masked Policy (Selects uniformly among valid collision-free actions)
  5. Classical Raster Metrology (Uniform geometric sampling across all 5 faces)

Evaluates over N=50 stochastic test episodes with domain randomization, collecting:
- Cumulative episodic reward
- Surface uncertainty reduction & final uncertainty
- Total CMM travel time (Chebyshev clearance norm)
- Metrology plane-fitting success rate (>= 4 touches on all 5 faces)
- Inspection efficiency (Uncertainty reduction % per second of motion)
- Spatial spread on faces (mean intra-face touch distance)
- Touch measurement deviation error (mm)
- Neural network value estimates V(s_t) and policy entropy H(pi)
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from cmm_env import CMMTouchPointEnv
from pointcloud_preprocessor import PreprocessedCandidateData, PointCloudPreprocessor
from sb3_contrib import MaskablePPO


class BaselineAgent:
    """Base interface for comparative metrology inspection agents."""
    def __init__(self, name: str):
        self.name = name

    def select_action(self, env: CMMTouchPointEnv, obs: np.ndarray, masks: np.ndarray) -> int:
        raise NotImplementedError


class RandomMaskedAgent(BaselineAgent):
    """Uniform random selection among collision-free and unvisited candidate points."""
    def __init__(self):
        super().__init__("Random Masked")

    def select_action(self, env: CMMTouchPointEnv, obs: np.ndarray, masks: np.ndarray) -> int:
        valid_indices = np.where(masks)[0]
        if len(valid_indices) == 0:
            return 0
        return int(np.random.choice(valid_indices))


class GreedyUncertaintyAgent(BaselineAgent):
    """Greedy Next-Best-Touch: selects valid point yielding highest one-step uncertainty drop."""
    def __init__(self):
        super().__init__("Greedy Uncertainty")

    def select_action(self, env: CMMTouchPointEnv, obs: np.ndarray, masks: np.ndarray) -> int:
        valid_indices = np.where(masks)[0]
        if len(valid_indices) == 0:
            return 0

        # Evaluate projected uncertainty reduction for all valid points
        best_action = valid_indices[0]
        best_reduction = -1.0

        for action in valid_indices:
            p_cand = env.coords[action]
            n_cand = env.normals[action]

            delta_p = env.coords - p_cand
            dist_sq = np.sum(delta_p * delta_p, axis=1)
            normal_dots = np.maximum(0.0, np.sum(env.normals * n_cand, axis=1))

            spatial_k = np.exp(-dist_sq / (2.0 * (env.length_scale ** 2)))
            normal_k = normal_dots ** 4
            weights = spatial_k * normal_k

            projected_reduction = np.sum(env.uncertainties * weights)
            if projected_reduction > best_reduction:
                best_reduction = projected_reduction
                best_action = action

        return int(best_action)


class NearestNeighborAgent(BaselineAgent):
    """Greedy TSP Travel-Time Minimizer: selects valid point closest to current probe."""
    def __init__(self):
        super().__init__("Nearest Neighbor")

    def select_action(self, env: CMMTouchPointEnv, obs: np.ndarray, masks: np.ndarray) -> int:
        valid_indices = np.where(masks)[0]
        if len(valid_indices) == 0:
            return 0

        best_action = valid_indices[0]
        min_travel_time = 1e9

        cloud_max_z = float(env.coords[:, 2].max())
        for action in valid_indices:
            p_cand = env.coords[action]
            n_cand = env.normals[action]

            t_move = env.math_engine.compute_travel_time(
                p_prev=env.probe_pos,
                p_curr=p_cand,
                n_curr=n_cand,
                cloud_max_z=cloud_max_z
            )
            if t_move < min_travel_time:
                min_travel_time = t_move
                best_action = action

        return int(best_action)


class RasterMetrologyAgent(BaselineAgent):
    """
    Standard Industrial Practice: 4 well-separated touchpoints per face,
    cycling through Top, North, South, East, West sequentially.
    """
    def __init__(self):
        super().__init__("Raster Grid")

    def select_action(self, env: CMMTouchPointEnv, obs: np.ndarray, masks: np.ndarray) -> int:
        valid_indices = np.where(masks)[0]
        if len(valid_indices) == 0:
            return 0

        # Current face target: target face that has fewer than 4 touches
        needed_faces = np.where(env.face_touch_counts < 4)[0]
        if len(needed_faces) > 0:
            target_face = needed_faces[0]
            face_cand = np.where((env.primitive_ids == target_face) & masks)[0]
            if len(face_cand) > 0:
                # Pick point maximizing distance to previous touches on this face
                prev_pts = env.face_touch_pts[target_face]
                if len(prev_pts) == 0:
                    return int(face_cand[0])
                else:
                    best_cand = face_cand[0]
                    max_d = -1.0
                    for c in face_cand:
                        d_min = min(np.linalg.norm(env.coords[c] - p) for p in prev_pts)
                        if d_min > max_d:
                            max_d = d_min
                            best_cand = c
                    return int(best_cand)

        # Fallback to valid candidate
        return int(valid_indices[0])


class RLMaskablePPOAgent(BaselineAgent):
    """Trained MaskablePPO Agent with value extraction and entropy logging."""
    def __init__(self, model_path: str):
        super().__init__("MaskablePPO (Ours)")
        self.model = MaskablePPO.load(model_path)

    def select_action(self, env: CMMTouchPointEnv, obs: np.ndarray, masks: np.ndarray) -> int:
        action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
        return int(action)

    def get_value_and_entropy(self, obs: np.ndarray, masks: np.ndarray) -> Tuple[float, float]:
        """Extract value estimate V(s) and policy entropy H(pi) from PyTorch actor-critic."""
        import torch
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        mask_tensor = torch.as_tensor(masks).bool().unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            # Value estimate
            value = self.model.policy.predict_values(obs_tensor)
            val_scalar = float(value.squeeze().cpu().item())

            # Distribution and entropy
            dist = self.model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            entropy = float(dist.entropy().squeeze().cpu().item())

        return val_scalar, entropy


def evaluate_agent(agent: BaselineAgent,
                   env: CMMTouchPointEnv,
                   num_episodes: int = 50,
                   base_seed: int = 1000) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run Monte Carlo evaluation of an agent across num_episodes.
    Returns:
        episode_metrics: List of metrics for each episode.
        summary_stats: Mean and std across all episodes.
        stepwise_telemetry: Detailed per-step trace for the first episode.
    """
    episodes = []
    stepwise_first_ep = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        terminated = False
        truncated = False
        ep_reward = 0.0
        step_idx = 0
        initial_unc = float(np.mean(env.uncertainties))

        is_first_ep = (ep == 0)

        while not (terminated or truncated):
            step_idx += 1
            masks = env.action_masks()

            # Record value & entropy if MaskablePPO
            val_est, entropy_est = 0.0, 0.0
            if isinstance(agent, RLMaskablePPOAgent):
                val_est, entropy_est = agent.get_value_and_entropy(obs, masks)

            prev_pos = env.probe_pos.copy()
            prev_unc = float(np.mean(env.uncertainties))

            action = agent.select_action(env, obs, masks)
            chosen_pt = env.coords[action].copy()
            chosen_n = env.normals[action].copy()
            chosen_face = int(env.primitive_ids[action])

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            curr_unc = float(np.mean(env.uncertainties))

            if is_first_ep:
                # Step reward breakdown
                w1, w2, w3, w4 = env.reward_weights
                # Uncertainty reduction
                r_unc = (prev_unc - curr_unc) * 100.0
                # Primitive bonus
                touches_on_face = env.face_touch_counts[chosen_face]
                r_prim = 3.0 if touches_on_face <= 4 else -1.5
                # Travel time
                t_step = info.get('total_travel_time', 0.0)

                stepwise_first_ep.append({
                    'step': step_idx,
                    'action': int(action),
                    'point': chosen_pt.tolist(),
                    'normal': chosen_n.tolist(),
                    'face_id': chosen_face,
                    'face_name': PointCloudPreprocessor.FACE_NAMES[chosen_face],
                    'reward': float(reward),
                    'remaining_uncertainty': float(curr_unc),
                    'step_unc_drop': float(prev_unc - curr_unc),
                    'total_travel_time': float(info.get('total_travel_time', 0.0)),
                    'value_estimate': val_est,
                    'policy_entropy': entropy_est
                })

        final_unc = float(np.mean(env.uncertainties))
        unc_drop = initial_unc - final_unc
        t_total = float(env.total_travel_time)
        face_counts = env.face_touch_counts.copy()
        success = 1.0 if np.all(face_counts >= 4) else 0.0
        efficiency = (unc_drop * 100.0) / max(t_total, 1e-4)  # % uncertainty drop per second

        # Calculate mean intra-face touchpoint spread
        spreads = []
        for f in range(5):
            pts = env.face_touch_pts[f]
            if len(pts) >= 2:
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        spreads.append(np.linalg.norm(pts[i] - pts[j]))
        mean_spread = float(np.mean(spreads)) if spreads else 0.0
        mean_err_mm = (float(np.mean(env.touch_errors)) * 1000.0) if env.touch_errors else 0.0

        episodes.append({
            'episode': ep + 1,
            'reward': float(ep_reward),
            'travel_time': t_total,
            'initial_uncertainty': float(initial_unc),
            'final_uncertainty': float(final_unc),
            'uncertainty_reduction': float(unc_drop),
            'uncertainty_reduction_pct': float(unc_drop * 100.0),
            'efficiency_pct_per_sec': float(efficiency),
            'success': float(success),
            'face_counts': face_counts.tolist(),
            'mean_spread_m': mean_spread,
            'mean_error_mm': mean_err_mm
        })

    # Summary statistics
    rewards = [e['reward'] for e in episodes]
    times = [e['travel_time'] for e in episodes]
    final_uncs = [e['final_uncertainty'] * 100.0 for e in episodes]
    drops = [e['uncertainty_reduction_pct'] for e in episodes]
    efficiencies = [e['efficiency_pct_per_sec'] for e in episodes]
    successes = [e['success'] for e in episodes]
    spreads = [e['mean_spread_m'] * 1000.0 for e in episodes]  # in mm
    errors = [e['mean_error_mm'] for e in episodes]

    summary = {
        'agent_name': agent.name,
        'episodes_evaluated': num_episodes,
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'travel_time_mean': float(np.mean(times)),
        'travel_time_std': float(np.std(times)),
        'final_uncertainty_mean_pct': float(np.mean(final_uncs)),
        'final_uncertainty_std_pct': float(np.std(final_uncs)),
        'uncertainty_reduction_mean_pct': float(np.mean(drops)),
        'uncertainty_reduction_std_pct': float(np.std(drops)),
        'efficiency_mean_pct_per_sec': float(np.mean(efficiencies)),
        'efficiency_std_pct_per_sec': float(np.std(efficiencies)),
        'success_rate_pct': float(np.mean(successes) * 100.0),
        'mean_spread_mm': float(np.mean(spreads)),
        'mean_error_mm': float(np.mean(errors)),
    }

    return episodes, summary, stepwise_first_ep


def main():
    parser = argparse.ArgumentParser(description="Academic Metrology Benchmarking Suite")
    parser.add_argument("--model", type=str, default="cmm_maskable_ppo.zip")
    parser.add_argument("--npz", type=str, default="candidates_256.npz")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="benchmark_output")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*95}")
    print("       COMPREHENSIVE CMM METROLOGY INSPECTION BENCHMARK & COMPARATIVE EVALUATION")
    print(f"{'='*95}")
    print(f"Candidates NPZ   : {args.npz}")
    print(f"Model Path       : {args.model}")
    print(f"Test Episodes    : {args.episodes}")
    print(f"Random Seed Base : {args.seed}")
    print(f"Output Directory : {args.output_dir}")
    print(f"{'='*95}\n")

    env = CMMTouchPointEnv(
        npz_path=args.npz,
        K=256,
        max_touches=20,
        domain_randomize=True
    )

    agents = [
        RLMaskablePPOAgent(model_path=args.model),
        GreedyUncertaintyAgent(),
        NearestNeighborAgent(),
        RasterMetrologyAgent(),
        RandomMaskedAgent(),
    ]

    all_summaries = []
    all_episodes_data = {}
    all_stepwise_data = {}

    for agent in agents:
        print(f"[RUNNING] Benchmarking: {agent.name} over {args.episodes} episodes...")
        t0 = time.time()
        episodes, summary, stepwise = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=args.episodes,
            base_seed=args.seed
        )
        elapsed = time.time() - t0
        print(f"  -> Completed in {elapsed:.2f}s | Success: {summary['success_rate_pct']:.1f}% | "
              f"Reward: {summary['reward_mean']:.1f} +/- {summary['reward_std']:.1f} | "
              f"Travel: {summary['travel_time_mean']:.2f}s | "
              f"Uncertainty Drop: {summary['uncertainty_reduction_mean_pct']:.1f}% | "
              f"Efficiency: {summary['efficiency_mean_pct_per_sec']:.2f} %/s")

        all_summaries.append(summary)
        all_episodes_data[agent.name] = episodes
        all_stepwise_data[agent.name] = stepwise

    # Print Formatted Benchmark Comparison Table
    print("\n" + "="*115)
    print(f"{'Method':<24} | {'Success':^9} | {'Reward':^16} | {'Travel Time (s)':^16} | {'Unc Drop (%)':^15} | {'Efficiency (%/s)':^16}")
    print("="*115)
    for s in all_summaries:
        rew_str = f"{s['reward_mean']:6.1f} ± {s['reward_std']:4.1f}"
        time_str = f"{s['travel_time_mean']:5.2f} ± {s['travel_time_std']:4.2f}"
        unc_str = f"{s['uncertainty_reduction_mean_pct']:5.1f} ± {s['uncertainty_reduction_std_pct']:4.1f}"
        eff_str = f"{s['efficiency_mean_pct_per_sec']:5.2f} ± {s['efficiency_std_pct_per_sec']:4.2f}"
        succ_str = f"{s['success_rate_pct']:5.1f}%"
        print(f"{s['agent_name']:<24} | {succ_str:^9} | {rew_str:^16} | {time_str:^16} | {unc_str:^15} | {eff_str:^16}")
    print("="*115 + "\n")

    # Export to CSV and JSON
    summary_df = pd.DataFrame(all_summaries)
    csv_summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_df.to_csv(csv_summary_path, index=False)
    print(f"[SAVED] Benchmark summary table exported to '{csv_summary_path}'")

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'summaries': all_summaries,
            'episodes': all_episodes_data,
            'stepwise': all_stepwise_data
        }, f, indent=4)
    print(f"[SAVED] Comprehensive benchmark raw dataset exported to '{json_path}'")


if __name__ == '__main__':
    main()
