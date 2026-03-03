"""
evaluate_ppo.py — PPO Agent Evaluation Script
Team Member 3 deliverable

Evaluates trained PPO agent against baselines across all workload modes.

Usage:
    python evaluate_ppo.py
"""

import numpy as np
import sys
import os

# Path fixes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))

from cloud_env import CloudEnv
from ppo_agent import PPOAgent
from baselines import StaticPolicy, ThresholdPolicy, AutoScalePolicy


def evaluate_episode(policy, env, deterministic=True, is_ppo=False):
    """
    Run one episode and collect metrics.
    
    Parameters
    ----------
    policy : PPOAgent or BasePolicy
        Agent/policy to evaluate
    env : CloudEnv
        Environment
    deterministic : bool
        For PPO: use argmax instead of sampling
    is_ppo : bool
        True if policy is PPOAgent
    
    Returns
    -------
    dict : Episode metrics
    """
    state = env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()
    
    total_reward = 0.0
    sla_violations = 0
    latencies = []
    vm_counts = []
    scaling_events = 0
    
    done = False
    while not done:
        if is_ppo:
            action, _, _ = policy.select_action(state, deterministic=deterministic)
        else:
            action = policy.select_action(state)
        
        state, reward, done, info = env.step(action)
        
        total_reward += reward
        sla_violations += info['sla_violation']
        latencies.append(info['latency'])
        vm_counts.append(info['n_active'])
        
        # Better FIX: Count an event if the actual VM count changed
        if len(vm_counts) > 1 and vm_counts[-1] != vm_counts[-2]:
            scaling_events += 1
    
    return {
        'total_reward': total_reward,
        'sla_violations': sla_violations,
        'sla_rate': sla_violations / len(latencies),
        'avg_latency': float(np.mean(latencies)),
        'p95_latency': float(np.percentile(latencies, 95)),
        'total_cost': float(np.sum(vm_counts)),
        'avg_vms': float(np.mean(vm_counts)),
        'n_scaling_events': scaling_events,
        'vm_variance': float(np.var(vm_counts)),
    }


def evaluate_policy(policy, workload_mode, n_runs=5, base_seed=42, is_ppo=False):
    """
    Evaluate policy over multiple runs.
    
    Returns
    -------
    dict : {metric: {'mean': float, 'std': float}}
    """
    results = []
    for i in range(n_runs):
        env = CloudEnv(workload_mode=workload_mode, seed=base_seed + i)
        result = evaluate_episode(policy, env, deterministic=True, is_ppo=is_ppo)
        results.append(result)
    
    # Compute mean and std
    metrics = list(results[0].keys())
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results]
        summary[m] = {'mean': np.mean(vals), 'std': np.std(vals)}
    
    return summary


def print_summary(name, summary):
    """Pretty print evaluation summary."""
    print(f"\n{'─'*60}")
    print(f"  Policy: {name}")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'Mean':>12}  {'±Std':>10}")
    print(f"  {'─'*48}")
    for metric, vals in summary.items():
        print(f"  {metric:<25} {vals['mean']:>12.3f}  {vals['std']:>10.3f}")


def print_comparison_table(results_dict, workload_mode):
    """
    Print comparison table for all policies.
    
    Parameters
    ----------
    results_dict : dict
        {policy_name: summary_dict}
    workload_mode : str
        Workload mode name
    """
    print(f"\n\n{'█'*70}")
    print(f"  COMPARISON TABLE — {workload_mode.upper()} WORKLOAD")
    print(f"{'█'*70}")
    
    # Key metrics to compare
    key_metrics = ['total_reward', 'sla_rate', 'avg_latency', 'total_cost', 'n_scaling_events']
    
    # Header
    policies = list(results_dict.keys())
    header = f"  {'Metric':<20}"
    for policy in policies:
        header += f" {policy:>15}"
    print(header)
    print(f"  {'─'*65}")
    
    # Rows
    for metric in key_metrics:
        row = f"  {metric:<20}"
        for policy in policies:
            mean = results_dict[policy][metric]['mean']
            row += f" {mean:>15.2f}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  PPO vs BASELINES EVALUATION")
    print("="*70)
    
    N_RUNS = 5
    BASE_SEED = 42
    WORKLOAD_MODES = ['sinusoidal', 'spike', 'poisson']
    
    # Define policies
    baselines = [
        ('Static', StaticPolicy(), False),
        ('Threshold', ThresholdPolicy(), False),
        ('AutoScale', AutoScalePolicy(), False),
    ]
    
    # Load trained PPO agent
    ppo_agent = PPOAgent(state_dim=8, action_dim=5)
    try:
        ppo_agent.load('ppo_agent.pkl')
        print("\n✓ Loaded trained PPO agent from ppo_agent.pkl")
        policies = baselines + [('PPO', ppo_agent, True)]
    except FileNotFoundError:
        print("\n✗ ppo_agent.pkl not found. Run ppo_agent.py first to train.")
        print("  Evaluating baselines only...\n")
        policies = baselines
    
    # Evaluate all policies on all workload modes
    for mode in WORKLOAD_MODES:
        print(f"\n\n{'█'*70}")
        print(f"  WORKLOAD MODE: {mode.upper()}")
        print(f"{'█'*70}")
        
        results = {}
        
        for name, policy, is_ppo in policies:
            print(f"\nEvaluating {name}...")
            summary = evaluate_policy(policy, mode, n_runs=N_RUNS, base_seed=BASE_SEED, is_ppo=is_ppo)
            results[name] = summary
            print_summary(name, summary)
        
        # Print comparison table
        print_comparison_table(results, mode)
    
    print("\n\n" + "="*70)
    print("  Evaluation Complete!")
    print("="*70 + "\n")
