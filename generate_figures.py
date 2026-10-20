#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Suite for CMM Next-Best-Touch RL Report.
Produces high-resolution (300 DPI), professionally styled figures using Seaborn and Matplotlib:
  - Figure 1: Learning Curves & Metrology Convergence Progression
  - Figure 2: PPO Policy & Value Function Optimization Diagnostics
  - Figure 3: Comparative Benchmarking against Metrology Baselines
  - Figure 4: Face Coverage & Plane-Fitting Feasibility Breakdown
  - Figure 5: Stepwise Inspection Dynamics & Reward Decomposition
  - Figure 6: Actor-Critic Value Function & Policy Entropy Evolution
  - Figure 7: 3D Spatial Touchpoint Distribution on Part Geometry
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
    'grid.color': '#dddddd',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

PALETTE = {
    'MaskablePPO (Ours)': '#1f77b4',       # Strong Blue
    'Greedy Uncertainty': '#ff7f0e',       # Amber Orange
    'Nearest Neighbor': '#2ca02c',         # Forest Green
    'Raster Grid': '#9467bd',              # Royal Purple
    'Random Masked': '#d62728',            # Crimson Red
}


def plot_fig1_learning_curves(episodes_csv: str, output_path: str):
    """Figure 1: Training Progression & Convergence Curves."""
    if not os.path.isfile(episodes_csv):
        print(f"[WARN] Skipping Fig 1: {episodes_csv} not found.")
        return

    df = pd.read_csv(episodes_csv)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)

    # 1. Episodic Return & Moving Average
    ax = axes[0, 0]
    ax.scatter(df['timestep'], df['reward'], alpha=0.15, color='#4a90e2', s=12, label='Episode Return')
    ax.plot(df['timestep'], df['moving_avg_reward'], color='#003f5c', linewidth=2.2, label='20-Ep Moving Average')
    ax.set_title('(A) Episodic Return Progression', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Total Episodic Reward')
    ax.legend(loc='lower right', frameon=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 2. Metrology Success Rate
    ax = axes[0, 1]
    ax.plot(df['timestep'], df['moving_avg_success'] * 100.0, color='#2f4b7c', linewidth=2.2)
    ax.axhline(100.0, color='#2ca02c', linestyle=':', alpha=0.8, label='100% Target')
    ax.set_title('(B) Metrology Plane-Fitting Success Rate (4 Pts/Face)', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower right', frameon=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 3. Final Remaining Uncertainty
    ax = axes[1, 0]
    unc_pct = df['remaining_uncertainty'] * 100.0
    unc_ma = unc_pct.rolling(window=20, min_periods=1).mean()
    ax.scatter(df['timestep'], unc_pct, alpha=0.15, color='#ff7c43', s=12, label='Raw Episode Final Unc')
    ax.plot(df['timestep'], unc_ma, color='#d45087', linewidth=2.2, label='20-Ep Moving Average')
    ax.axhline(68.6, color='black', linestyle='--', alpha=0.5, label='Initial Prior Unc (68.6%)')
    ax.set_title('(C) Remaining Surface Uncertainty Contraction', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Remaining Uncertainty (%)')
    ax.legend(loc='upper right', frameon=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 4. Total CMM Inspection Travel Time
    ax = axes[1, 1]
    time_ma = df['travel_time'].rolling(window=20, min_periods=1).mean()
    ax.scatter(df['timestep'], df['travel_time'], alpha=0.15, color='#665191', s=12, label='Episode Travel Time')
    ax.plot(df['timestep'], time_ma, color='#a05195', linewidth=2.2, label='20-Ep Moving Average')
    ax.set_title('(D) Total CMM Clearance-Box Travel Time', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Chebyshev Motion Duration (s)')
    ax.legend(loc='upper right', frameon=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    plt.suptitle('Figure 1: MaskablePPO Training Dynamics & Metrology Convergence (40,960 Steps)', fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 1 at '{output_path}'")


def plot_fig2_optimization_diagnostics(updates_csv: str, output_path: str):
    """Figure 2: PPO Policy & Value Function Optimization Diagnostics."""
    if not os.path.isfile(updates_csv):
        print(f"[WARN] Skipping Fig 2: {updates_csv} not found.")
        return

    df = pd.read_csv(updates_csv)
    if len(df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)

    # 1. Policy Gradient Loss
    ax = axes[0, 0]
    ax.plot(df['timestep'], df['policy_loss'], color='#0072b2', marker='o', markersize=4, linewidth=2.0)
    ax.set_title('(A) Masked PPO Clipped Surrogate Loss ($L^{CLIP}$)', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Policy Gradient Loss')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 2. Value Function MSE Loss
    ax = axes[0, 1]
    ax.plot(df['timestep'], df['value_loss'], color='#d55e00', marker='s', markersize=4, linewidth=2.0)
    ax.set_title('(B) Critic Value Function Mean Squared Error ($L^V$)', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Value Loss')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 3. Policy Entropy Decay
    ax = axes[1, 0]
    ax.plot(df['timestep'], df['entropy_loss'], color='#cc79a7', marker='^', markersize=4, linewidth=2.0)
    ax.set_title('(C) Policy Action Entropy ($S[\\pi]$)', fontweight='bold')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Entropy Bonus Term')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    # 4. Approximate KL Divergence & Explained Variance
    ax = axes[1, 1]
    color_kl = '#009e73'
    ax.plot(df['timestep'], df['approx_kl'], color=color_kl, marker='d', markersize=4, linewidth=2.0, label='Approx KL')
    ax.set_xlabel('Environment Timesteps')
    ax.set_ylabel('Approximate KL Divergence', color=color_kl)
    ax.tick_params(axis='y', labelcolor=color_kl)
    ax.set_title('(D) Approximate KL Divergence & Value Explained Variance', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1000)}k"))

    ax2 = ax.twinx()
    color_ev = '#56b4e9'
    ax2.plot(df['timestep'], df['explained_variance'], color=color_ev, linestyle='--', marker='x', markersize=5, linewidth=2.0, label='Explained Var')
    ax2.set_ylabel('Explained Variance', color=color_ev)
    ax2.tick_params(axis='y', labelcolor=color_ev)
    ax2.set_ylim(-0.1, 1.05)
    ax2.grid(False)

    plt.suptitle('Figure 2: MaskablePPO Actor-Critic Neural Optimization Diagnostics', fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 2 at '{output_path}'")


def plot_fig3_baseline_comparison(benchmark_json: str, output_path: str):
    """Figure 3: Comparative Benchmarking against Baselines."""
    if not os.path.isfile(benchmark_json):
        print(f"[WARN] Skipping Fig 3: {benchmark_json} not found.")
        return

    with open(benchmark_json, 'r') as f:
        data = json.load(f)

    episodes_dict = data['episodes']
    # Build a flat dataframe of episodes
    rows = []
    for agent_name, eps in episodes_dict.items():
        for e in eps:
            rows.append({
                'Agent': agent_name,
                'Reward': e['reward'],
                'TravelTime': e['travel_time'],
                'UncertaintyDrop': e['uncertainty_reduction_pct'],
                'Efficiency': e['efficiency_pct_per_sec'],
                'Success': e['success'] * 100.0,
                'FinalUnc': e['final_uncertainty'] * 100.0
            })
    df = pd.DataFrame(rows)

    order = ['MaskablePPO (Ours)', 'Greedy Uncertainty', 'Nearest Neighbor', 'Raster Grid', 'Random Masked']
    palette = [PALETTE.get(k, '#333333') for k in order]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

    # 1. Total Episodic Reward (Boxplot + Stripplot)
    ax = axes[0, 0]
    sns.boxplot(data=df, x='Agent', y='Reward', order=order, palette=palette, ax=ax, width=0.5, fliersize=0)
    sns.stripplot(data=df, x='Agent', y='Reward', order=order, color='black', alpha=0.3, jitter=0.2, size=4, ax=ax)
    ax.set_title('(A) Cumulative Episodic Reward', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(order, rotation=15, ha='right')
    ax.set_ylabel('Reward Return')

    # 2. Total Uncertainty Drop (%)
    ax = axes[0, 1]
    sns.barplot(data=df, x='Agent', y='UncertaintyDrop', order=order, palette=palette, ax=ax, capsize=0.1, err_kws={'linewidth': 1.5})
    ax.set_title('(B) Surface Uncertainty Reduction (%)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(order, rotation=15, ha='right')
    ax.set_ylabel('Uncertainty Drop (\\Delta \\sigma \\times 100)')

    # 3. Total Travel Time (s)
    ax = axes[1, 0]
    sns.barplot(data=df, x='Agent', y='TravelTime', order=order, palette=palette, ax=ax, capsize=0.1, err_kws={'linewidth': 1.5})
    ax.set_title('(C) Total CMM Travel Time (Chebyshev Clearance Path)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(order, rotation=15, ha='right')
    ax.set_ylabel('Travel Time (seconds)')

    # 4. Inspection Efficiency (% drop / second)
    ax = axes[1, 1]
    sns.barplot(data=df, x='Agent', y='Efficiency', order=order, palette=palette, ax=ax, capsize=0.1, err_kws={'linewidth': 1.5})
    ax.set_title('(D) Metrology Inspection Efficiency (% / sec)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(order, rotation=15, ha='right')
    ax.set_ylabel('Uncertainty Reduction / Second (%/s)')

    plt.suptitle('Figure 3: Metrology Performance Benchmark across 50 Monte Carlo Evaluation Runs', fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 3 at '{output_path}'")


def plot_fig4_face_coverage(benchmark_json: str, output_path: str):
    """Figure 4: Face Coverage Breakdown across the 5 Workpiece Faces."""
    if not os.path.isfile(benchmark_json):
        print(f"[WARN] Skipping Fig 4: {benchmark_json} not found.")
        return

    with open(benchmark_json, 'r') as f:
        data = json.load(f)

    episodes_dict = data['episodes']
    face_names = ['Top (+Z)', 'North (+X)', 'South (-X)', 'East (+Y)', 'West (-Y)']
    order = ['MaskablePPO (Ours)', 'Greedy Uncertainty', 'Nearest Neighbor', 'Raster Grid', 'Random Masked']

    face_records = []
    for agent_name in order:
        if agent_name not in episodes_dict:
            continue
        eps = episodes_dict[agent_name]
        all_counts = [e['face_counts'] for e in eps]
        mean_counts = np.mean(all_counts, axis=0)
        for f_idx, count in enumerate(mean_counts):
            face_records.append({
                'Agent': agent_name,
                'Face': face_names[f_idx],
                'Touches': count
            })

    df_face = pd.DataFrame(face_records)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    palette = [PALETTE.get(k, '#333333') for k in order]

    sns.barplot(data=df_face, x='Face', y='Touches', hue='Agent', palette=palette, ax=ax)
    ax.axhline(4.0, color='red', linestyle='--', linewidth=2.0, label='Metrology Threshold (>= 4 pts/face)')
    ax.set_title('Figure 4: Mean Touchpoint Distribution Across 5 Workpiece Faces (N=50 Episodes)', fontweight='bold')
    ax.set_xlabel('CAD Workpiece Face')
    ax.set_ylabel('Mean Touch Count (Budget = 20)')
    ax.set_ylim(0, 10)
    ax.legend(title='Inspection Strategy', loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 4 at '{output_path}'")


def plot_fig5_stepwise_dynamics(benchmark_json: str, output_path: str):
    """Figure 5: Step-by-Step Inspection Dynamics for MaskablePPO."""
    if not os.path.isfile(benchmark_json):
        print(f"[WARN] Skipping Fig 5: {benchmark_json} not found.")
        return

    with open(benchmark_json, 'r') as f:
        data = json.load(f)

    stepwise_dict = data.get('stepwise', {})
    rl_steps = stepwise_dict.get('MaskablePPO (Ours)', [])
    if not rl_steps:
        print("[WARN] Skipping Fig 5: No RL stepwise trace found.")
        return

    df_steps = pd.DataFrame(rl_steps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, dpi=300)

    # 1. Step Reward
    ax = axes[0]
    ax.plot(df_steps['step'], df_steps['reward'], color='#1f77b4', marker='o', linewidth=2.2)
    ax.set_title('(A) Immediate Step Reward $R(s_t, a_t)$ Across 20 Touch Budget', fontweight='bold')
    ax.set_ylabel('Reward')

    # 2. Cumulative Uncertainty Drop
    ax = axes[1]
    unc_pct = df_steps['remaining_uncertainty'] * 100.0
    ax.plot(df_steps['step'], unc_pct, color='#d62728', marker='s', linewidth=2.2, label='Remaining Uncertainty')
    ax.set_title('(B) Surface Uncertainty Contraction Over Inspection Sequence', fontweight='bold')
    ax.set_ylabel('Uncertainty (%)')
    ax.legend(loc='upper right')

    # 3. Face Transition Sequence
    ax = axes[2]
    face_names = ['Top (+Z)', 'North (+X)', 'South (-X)', 'East (+Y)', 'West (-Y)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    face_colors = [colors[int(f)] for f in df_steps['face_id']]

    bars = ax.bar(df_steps['step'], [1]*len(df_steps), color=face_colors, edgecolor='black', alpha=0.85)
    ax.set_title('(C) Autonomous CMM Workpiece Face Targeting Sequence', fontweight='bold')
    ax.set_xlabel('Touchpoint Step ($t = 1 \\dots 20$)')
    ax.set_yticks([])
    ax.set_xticks(range(1, 21))

    # Add custom legend for faces
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[i], label=f"Face {i}: {face_names[i]}") for i in range(5)]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True)

    plt.suptitle('Figure 5: Step-by-Step Trajectory & Targeting Sequence of Trained RL Policy', fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 5 at '{output_path}'")


def plot_fig6_value_entropy(benchmark_json: str, output_path: str):
    """Figure 6: Value Function V(s) and Policy Entropy H(pi) Across Episode Steps."""
    if not os.path.isfile(benchmark_json):
        print(f"[WARN] Skipping Fig 6: {benchmark_json} not found.")
        return

    with open(benchmark_json, 'r') as f:
        data = json.load(f)

    stepwise_dict = data.get('stepwise', {})
    rl_steps = stepwise_dict.get('MaskablePPO (Ours)', [])
    if not rl_steps:
        return

    df_steps = pd.DataFrame(rl_steps)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, dpi=300)

    # 1. Critic State-Value Estimate V(s)
    ax = axes[0]
    ax.plot(df_steps['step'], df_steps['value_estimate'], color='#0072b2', marker='o', linewidth=2.2)
    ax.set_title('(A) Neural Critic Value Function Prediction $V_\\phi(s_t)$', fontweight='bold')
    ax.set_ylabel('Predicted Return $V(s_t)$')

    # 2. Policy Action Entropy H(pi)
    ax = axes[1]
    ax.plot(df_steps['step'], df_steps['policy_entropy'], color='#e69f00', marker='^', linewidth=2.2)
    ax.set_title('(B) Action Masked Policy Entropy $H[\\pi_\\theta(\\cdot \\mid s_t)]$', fontweight='bold')
    ax.set_xlabel('Touchpoint Step ($t = 1 \\dots 20$)')
    ax.set_ylabel('Entropy (nats)')
    ax.set_xticks(range(1, 21))

    plt.suptitle('Figure 6: Actor-Critic Internal Representation Evolution Over Episode Horizon', fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 6 at '{output_path}'")


def plot_fig7_spatial_distribution(benchmark_json: str, npz_path: str, output_path: str):
    """Figure 7: 3D / 2D Spatial Touchpoint Distribution on Part Geometry."""
    if not os.path.isfile(benchmark_json) or not os.path.isfile(npz_path):
        print(f"[WARN] Skipping Fig 7: files not found.")
        return

    with open(benchmark_json, 'r') as f:
        data = json.load(f)

    stepwise_dict = data.get('stepwise', {})
    rl_steps = stepwise_dict.get('MaskablePPO (Ours)', [])
    if not rl_steps:
        return

    npz_data = np.load(npz_path)
    cand_coords = npz_data['coords']
    cand_prim = npz_data['primitive_ids']

    # 3D Plot
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot candidate cloud (faded grey)
    ax.scatter(cand_coords[:, 0]*1000, cand_coords[:, 1]*1000, cand_coords[:, 2]*1000,
               c='#cccccc', alpha=0.35, s=15, label='K=256 Candidates')

    # Plot selected touchpoints colored by step order
    pts = np.array([s['point'] for s in rl_steps]) * 1000  # in mm
    steps = [s['step'] for s in rl_steps]

    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=steps, cmap='plasma', s=90, edgecolors='black', linewidth=1.2,
                         label='Planned Touchpoints')

    # Plot trajectory path
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#003f5c', linestyle='--', linewidth=1.5, alpha=0.7)

    # Number each touchpoint
    for i, s in enumerate(steps):
        ax.text(pts[i, 0] + 2, pts[i, 1] + 2, pts[i, 2] + 2, str(s), fontsize=9, fontweight='bold')

    ax.set_title('Figure 7: 3D Spatial Touchpoint Distribution and Inspection Sequence\nPlanned by MaskablePPO on CAD Workpiece', fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(loc='upper right')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Touch Execution Order (1 to 20)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Generated Figure 7 at '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures for CMM RL report")
    parser.add_argument("--episodes-csv", type=str, default="training_logs/training_episodes.csv")
    parser.add_argument("--updates-csv", type=str, default="training_logs/training_updates.csv")
    parser.add_argument("--benchmark-json", type=str, default="benchmark_output/benchmark_results.json")
    parser.add_argument("--npz", type=str, default="candidates_256.npz")
    parser.add_argument("--fig-dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    print("\n" + "="*80)
    print("      GENERATING PUBLICATION-READY FIGURES FOR RL CMM REPORT")
    print("="*80)

    plot_fig1_learning_curves(args.episodes_csv, os.path.join(args.fig_dir, "fig1_learning_curves.png"))
    plot_fig2_optimization_diagnostics(args.updates_csv, os.path.join(args.fig_dir, "fig2_ppo_optimization_diagnostics.png"))
    plot_fig3_baseline_comparison(args.benchmark_json, os.path.join(args.fig_dir, "fig3_baseline_comparison.png"))
    plot_fig4_face_coverage(args.benchmark_json, os.path.join(args.fig_dir, "fig4_face_coverage_distribution.png"))
    plot_fig5_stepwise_dynamics(args.benchmark_json, os.path.join(args.fig_dir, "fig5_stepwise_inspection_dynamics.png"))
    plot_fig6_value_entropy(args.benchmark_json, os.path.join(args.fig_dir, "fig6_value_entropy_progression.png"))
    plot_fig7_spatial_distribution(args.benchmark_json, args.npz, os.path.join(args.fig_dir, "fig7_spatial_touch_distribution.png"))

    print("\n[COMPLETE] All 7 publication figures successfully exported to:", args.fig_dir)


if __name__ == '__main__':
    main()
