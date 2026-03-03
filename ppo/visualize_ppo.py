"""
visualize_ppo.py — PPO Training and Performance Visualization
Team Member 3 deliverable

Generates plots for:
  1. Training curves (episode rewards)
  2. Policy behavior visualization (VM scaling over time)
  3. Comparison plots (PPO vs baselines)

Usage:
    python visualize_ppo.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Path fixes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))

from cloud_env import CloudEnv, ACTION_MAP
from ppo_agent import PPOAgent
from baselines import ThresholdPolicy, AutoScalePolicy


def plot_training_curve(agent, save_path='ppo_training_curve.png'):
    """Plot episode rewards over training."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    episodes = range(1, len(agent.episode_rewards) + 1)
    
    # Raw rewards
    ax1.plot(episodes, agent.episode_rewards, alpha=0.3, color='steelblue', label='Raw')
    
    # Moving average
    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(episodes)+1), moving_avg, color='darkblue', linewidth=2, label=f'{window}-episode avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training Progress — Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ax2.plot(episodes, agent.episode_lengths, alpha=0.5, color='coral')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths (should be constant at 500)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_policy_behavior(policy, env, is_ppo=True, policy_name='PPO', save_path='ppo_behavior.png'):
    """
    Visualize policy decisions over one episode.
    Shows: arrival rate, active VMs, queue, utilization, actions.
    """
    state = env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()
    
    # Data collection
    steps = []
    arrival_rates = []
    active_vms = []
    booting_vms = []
    queue_lengths = []
    utilizations = []
    actions_taken = []
    latencies = []
    
    done = False
    step = 0
    while not done:
        if is_ppo:
            action, _, _ = policy.select_action(state, deterministic=True)
        else:
            action = policy.select_action(state)
        
        state, reward, done, info = env.step(action)
        
        steps.append(step)
        arrival_rates.append(info['lambda_true'])
        active_vms.append(info['n_active'])
        booting_vms.append(info['n_booting'])
        queue_lengths.append(info['queue'])
        utilizations.append(info['utilization'])
        actions_taken.append(ACTION_MAP[action])
        latencies.append(info['latency'])
        
        step += 1
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle(f'{policy_name} Policy Behavior — {env.workload_mode.capitalize()} Workload', fontsize=14, fontweight='bold')
    
    # 1. Arrival rate
    axes[0].plot(steps, arrival_rates, color='purple', linewidth=1.5)
    axes[0].set_ylabel('Arrival Rate\n(req/step)', fontsize=10)
    axes[0].set_title('Workload Pattern')
    axes[0].grid(True, alpha=0.3)
    
    # 2. VM counts (stacked)
    axes[1].fill_between(steps, 0, active_vms, label='Active VMs', color='green', alpha=0.6)
    axes[1].fill_between(steps, active_vms, np.array(active_vms) + np.array(booting_vms), 
                         label='Booting VMs', color='orange', alpha=0.6)
    axes[1].set_ylabel('VM Count', fontsize=10)
    axes[1].set_title('Resource Provisioning')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Queue length
    axes[2].fill_between(steps, queue_lengths, color='red', alpha=0.5)
    axes[2].set_ylabel('Queue Length\n(requests)', fontsize=10)
    axes[2].set_title('Request Queue')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Utilization + latency
    ax_util = axes[3]
    ax_lat = ax_util.twinx()
    
    ax_util.plot(steps, utilizations, color='steelblue', linewidth=1.5, label='Utilization')
    ax_util.axhline(y=0.8, color='orange', linestyle='--', linewidth=1, label='Target (80%)')
    ax_util.set_ylabel('Utilization', fontsize=10, color='steelblue')
    ax_util.tick_params(axis='y', labelcolor='steelblue')
    ax_util.set_ylim(0, max(2.0, max(utilizations)))
    
    ax_lat.plot(steps, latencies, color='darkred', linewidth=1, alpha=0.7, label='Latency')
    ax_lat.axhline(y=5.0, color='red', linestyle='--', linewidth=1, label='SLA threshold')
    ax_lat.set_ylabel('Latency', fontsize=10, color='darkred')
    ax_lat.tick_params(axis='y', labelcolor='darkred')
    ax_lat.set_ylim(0, min(20, max(latencies) * 1.1))
    
    ax_util.set_title('Utilization & Latency')
    ax_util.grid(True, alpha=0.3)
    ax_util.legend(loc='upper left')
    ax_lat.legend(loc='upper right')
    
    # 5. Actions (bar chart)
    colors = ['red' if a < 0 else 'green' if a > 0 else 'gray' for a in actions_taken]
    axes[4].bar(steps, actions_taken, color=colors, alpha=0.7, width=1.0)
    axes[4].axhline(y=0, color='black', linewidth=0.8)
    axes[4].set_ylabel('Action\n(VM Δ)', fontsize=10)
    axes[4].set_xlabel('Time Step', fontsize=10)
    axes[4].set_title('Scaling Actions (Green=Scale Up, Red=Scale Down)')
    axes[4].set_ylim(-2.5, 2.5)
    axes[4].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


def plot_comparison(workload_mode='sinusoidal', save_path='ppo_comparison.png'):
    """
    Compare PPO vs baselines on key metrics (bar chart).
    """
    from evaluate_ppo import evaluate_policy
    
    # Load PPO agent
    ppo_agent = PPOAgent(state_dim=8, action_dim=5)
    try:
        ppo_agent.load('ppo_agent.pkl')
    except FileNotFoundError:
        print("Error: ppo_agent.pkl not found. Train agent first.")
        return
    
    # Evaluate all policies
    policies = [
        ('Threshold', ThresholdPolicy(), False),
        ('AutoScale', AutoScalePolicy(), False),
        ('PPO', ppo_agent, True),
    ]
    
    results = {}
    for name, policy, is_ppo in policies:
        print(f"Evaluating {name} on {workload_mode}...")
        summary = evaluate_policy(policy, workload_mode, n_runs=5, is_ppo=is_ppo)
        results[name] = summary
    
    # Plot comparison
    metrics = ['total_reward', 'sla_rate', 'avg_latency', 'total_cost']
    metric_labels = ['Total Reward', 'SLA Violation Rate', 'Avg Latency', 'Total Cost']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle(f'PPO vs Baselines — {workload_mode.capitalize()} Workload', fontsize=14, fontweight='bold')
    
    policy_names = list(results.keys())
    x = np.arange(len(policy_names))
    width = 0.6
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [results[p][metric]['mean'] for p in policy_names]
        stds = [results[p][metric]['std'] for p in policy_names]
        
        # Color code: PPO is highlighted
        colors = ['steelblue' if p != 'PPO' else 'forestgreen' for p in policy_names]
        
        axes[i].bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
        axes[i].set_ylabel(label, fontsize=11)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(policy_names, fontsize=10)
        axes[i].set_title(label)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Annotate values on bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            axes[i].text(j, mean + std, f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  PPO Visualization Suite")
    print("="*60)
    
    # Load trained agent
    agent = PPOAgent(state_dim=8, action_dim=5)
    try:
        agent.load('ppo_agent.pkl')
        print("\n✓ Loaded trained PPO agent")
    except FileNotFoundError:
        print("\n✗ ppo_agent.pkl not found. Train agent first with ppo_agent.py")
        exit(1)
    
    # 1. Training curve
    print("\n1. Plotting training curve...")
    plot_training_curve(agent, save_path='ppo_training_curve.png')
    
    # 2. Policy behavior on all workload modes
    print("\n2. Plotting policy behavior...")
    for mode in ['sinusoidal', 'spike', 'poisson']:
        env = CloudEnv(workload_mode=mode, seed=42)
        plot_policy_behavior(agent, env, is_ppo=True, policy_name='PPO', 
                            save_path=f'ppo_behavior_{mode}.png')
    
    # 3. Comparison plots
    print("\n3. Plotting comparison with baselines...")
    for mode in ['sinusoidal', 'spike', 'poisson']:
        plot_comparison(workload_mode=mode, save_path=f'ppo_comparison_{mode}.png')
    
    print("\n" + "="*60)
    print("  All visualizations complete!")
    print("="*60 + "\n")
