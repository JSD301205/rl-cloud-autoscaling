"""
evaluate_q_learning.py — Q-Learning Agent Evaluation Script
Team Member 3 deliverable

Evaluates trained tabular Q-learning agent against baselines across all workload modes.

Usage:
    python evaluate_q_learning.py
"""

import numpy as np
import sys
import os

# Path fixes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))

from cloud_env import CloudEnv
from q_learning_agent import QLearningAgent
from baselines import StaticPolicy, ThresholdPolicy, AutoScalePolicy


def evaluate_episode(policy, env, deterministic=True, is_q_learning=False):
    """
    Run one episode and collect metrics.

    Parameters
    ----------
    policy : QLearningAgent or BasePolicy
        Agent/policy to evaluate
    env : CloudEnv
        Environment
    deterministic : bool
        For Q-learning: use greedy action selection
    is_q_learning : bool
        True if policy is QLearningAgent

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
        if is_q_learning:
            action = policy.select_action(state, deterministic=deterministic)
        else:
            action = policy.select_action(state)

        state, reward, done, info = env.step(action)

        total_reward += reward
        sla_violations += info['sla_violation']
        latencies.append(info['latency'])
        vm_counts.append(info['n_active'])

        # Count an event when actual VM count changed
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


def evaluate_policy(policy, workload_mode, n_runs=5, base_seed=42, is_q_learning=False):
    """
    Evaluate policy over multiple runs.

    Returns
    -------
    dict : {metric: {'mean': float, 'std': float}}
    """
    results = []
    for i in range(n_runs):
        env = CloudEnv(workload_mode=workload_mode, seed=base_seed + i)
        result = evaluate_episode(policy, env, deterministic=True, is_q_learning=is_q_learning)
        results.append(result)

    metrics = list(results[0].keys())
    summary = {}
    for metric in metrics:
        vals = [r[metric] for r in results]
        summary[metric] = {'mean': np.mean(vals), 'std': np.std(vals)}

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

    key_metrics = ['total_reward', 'sla_rate', 'avg_latency', 'total_cost', 'n_scaling_events']

    policies = list(results_dict.keys())
    header = f"  {'Metric':<20}"
    for policy in policies:
        header += f" {policy:>15}"
    print(header)
    print(f"  {'─'*65}")

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
    print("  Q-LEARNING vs BASELINES EVALUATION")
    print("="*70)

    N_RUNS = 5
    BASE_SEED = 42
    WORKLOAD_MODES = ['sinusoidal', 'spike', 'poisson']

    baselines = [
        ('Static', StaticPolicy(), False),
        ('Threshold', ThresholdPolicy(), False),
        ('AutoScale', AutoScalePolicy(), False),
    ]

    # Load trained Q-learning agent
    q_agent = QLearningAgent(state_dim=2, action_dim=5, n_bins=5)
    try:
        q_agent.load('q_learning_agent.pkl')
        print("\n✓ Loaded trained Q-learning agent from q_learning_agent.pkl")
        policies = baselines + [('Q-Learning', q_agent, True)]
    except FileNotFoundError:
        print("\n✗ q_learning_agent.pkl not found. Run q_learning_agent.py first to train.")
        print("  Evaluating baselines only...\n")
        policies = baselines

    for mode in WORKLOAD_MODES:
        print(f"\n\n{'█'*70}")
        print(f"  WORKLOAD MODE: {mode.upper()}")
        print(f"{'█'*70}")

        results = {}

        for name, policy, is_q_learning in policies:
            print(f"\nEvaluating {name}...")
            summary = evaluate_policy(
                policy,
                mode,
                n_runs=N_RUNS,
                base_seed=BASE_SEED,
                is_q_learning=is_q_learning,
            )
            results[name] = summary
            print_summary(name, summary)

        print_comparison_table(results, mode)

    print("\n\n" + "="*70)
    print("  Evaluation Complete!")
    print("="*70 + "\n")
