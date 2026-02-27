"""
baselines.py — Classical Autoscaling Baselines
Team Member 2 deliverable

Implements 3 baseline autoscaling strategies:
  1. StaticPolicy         — fixed VM count, never scales
  2. ThresholdPolicy      — HPA-style CPU utilization threshold scaling
  3. AutoScalePolicy      — predictive scaling using recent arrival rate trend

All policies share the same interface:
    policy = StaticPolicy()
    action = policy.select_action(state)   # returns int in [0..4]

Action index → VM delta:
    0 → -2,  1 → -1,  2 → 0,  3 → +1,  4 → +2
"""

import numpy as np
import sys
import os

# ── Path fix so we can import CloudEnv from ../env ──────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
from cloud_env import CloudEnv, ACTION_MAP, MU

# Reverse map: delta → action index
DELTA_TO_ACTION = {v: k for k, v in ACTION_MAP.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────
class BasePolicy:
    """All policies inherit from this. Enforces the select_action interface."""

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self):
        """Called at the start of each episode. Override if needed."""
        pass

    def _clamp_delta(self, delta: int) -> int:
        """Clamp delta to valid action range [-2, +2] and return action index."""
        delta = int(np.clip(delta, -2, 2))
        return DELTA_TO_ACTION[delta]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Static Policy
# ─────────────────────────────────────────────────────────────────────────────
class StaticPolicy(BasePolicy):
    """
    Never scales. Keeps VM count fixed at whatever the environment starts with.
    Action is always 0 (no-op).

    This is the weakest baseline — shows the cost of no autoscaling.
    """

    def __init__(self):
        self.name = 'Static'

    def select_action(self, state: np.ndarray) -> int:
        return DELTA_TO_ACTION[0]   # always no-op


# ─────────────────────────────────────────────────────────────────────────────
# 2. Threshold Policy  (HPA-style)
# ─────────────────────────────────────────────────────────────────────────────
class ThresholdPolicy(BasePolicy):
    """
    Kubernetes HPA-style reactive scaling based on CPU utilization.

    Rules:
      - If utilization > scale_up_thresh   → scale up
      - If utilization < scale_down_thresh → scale down
      - Otherwise                          → no change

    Uses a cooldown window to avoid oscillation (thrashing).

    State index mapping (from cloud_env.py):
      state[3] = utilization  (u_t = λ_t / (n_t * μ))
      state[4] = n_active
    """

    def __init__(
        self,
        scale_up_thresh: float   = 0.80,   # scale up  if util > 80%
        scale_down_thresh: float = 0.30,   # scale down if util < 30%
        cooldown_steps: int      = 5,      # min steps between scaling events
    ):
        self.name             = 'Threshold (HPA)'
        self.scale_up_thresh  = scale_up_thresh
        self.scale_down_thresh= scale_down_thresh
        self.cooldown_steps   = cooldown_steps
        self._cooldown_timer  = 0

    def reset(self):
        self._cooldown_timer = 0

    def select_action(self, state: np.ndarray) -> int:
        utilization = float(state[3])

        # Tick cooldown
        if self._cooldown_timer > 0:
            self._cooldown_timer -= 1
            return DELTA_TO_ACTION[0]   # wait out cooldown

        if utilization > self.scale_up_thresh:
            # Scale up by 1 for moderate overload, 2 for heavy overload
            delta = 2 if utilization > 1.2 else 1
            self._cooldown_timer = self.cooldown_steps
            return self._clamp_delta(delta)

        elif utilization < self.scale_down_thresh:
            delta = -1
            self._cooldown_timer = self.cooldown_steps
            return self._clamp_delta(delta)

        return DELTA_TO_ACTION[0]   # no change


# ─────────────────────────────────────────────────────────────────────────────
# 3. AutoScale Policy  (Predictive)
# ─────────────────────────────────────────────────────────────────────────────
class AutoScalePolicy(BasePolicy):
    """
    Predictive scaling inspired by Gandhi et al. AutoScale (2012).

    Strategy:
      1. Use the 3 delayed arrival rates in state to estimate a trend.
      2. Project demand d steps ahead (accounting for boot delay).
      3. Compute ideal VM count = ceil(predicted_demand / μ).
      4. Scale toward ideal VM count.

    This is smarter than threshold — it anticipates load instead of reacting.

    State index mapping:
      state[0] = λ̃_t     (delayed arrival rate, most recent observed)
      state[1] = λ̃_{t-1}
      state[2] = λ̃_{t-2}
      state[4] = n_active
      state[6] = n_booting
    """

    def __init__(
        self,
        lookahead_steps: int = 3,    # match boot delay (predict 3 steps ahead)
        safety_margin: float = 1.15, # provision 15% extra headroom
        cooldown_steps: int  = 3,
    ):
        self.name            = 'AutoScale (Predictive)'
        self.lookahead       = lookahead_steps
        self.safety_margin   = safety_margin
        self.cooldown_steps  = cooldown_steps
        self._cooldown_timer = 0

    def reset(self):
        self._cooldown_timer = 0

    def select_action(self, state: np.ndarray) -> int:
        lam_0   = float(state[0])    # λ̃_t
        lam_1   = float(state[1])    # λ̃_{t-1}
        lam_2   = float(state[2])    # λ̃_{t-2}
        n_active  = float(state[4])
        n_booting = float(state[6])

        # ── Step 1: Estimate trend (linear regression on 3 points) ───────
        # Use simple finite difference: slope = (lam_0 - lam_2) / 2
        slope = (lam_0 - lam_2) / 2.0

        # ── Step 2: Project demand at (t + lookahead) ────────────────────
        predicted_demand = lam_0 + slope * self.lookahead
        predicted_demand = max(0.0, predicted_demand)

        # ── Step 3: Ideal VM count with safety margin ────────────────────
        ideal_vms = int(np.ceil(predicted_demand * self.safety_margin / MU))
        ideal_vms = int(np.clip(ideal_vms, 1, 20))

        # ── Step 4: Current effective capacity (active + booting) ─────────
        effective_vms = int(n_active + n_booting)

        # ── Step 5: Cooldown check ────────────────────────────────────────
        if self._cooldown_timer > 0:
            self._cooldown_timer -= 1
            return DELTA_TO_ACTION[0]

        # ── Step 6: Compute delta ─────────────────────────────────────────
        raw_delta = ideal_vms - int(n_active)

        if raw_delta == 0:
            return DELTA_TO_ACTION[0]

        # Clamp delta to [-2, +2]
        delta = int(np.clip(raw_delta, -2, 2))
        self._cooldown_timer = self.cooldown_steps
        return self._clamp_delta(delta)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(policy, workload_mode: str, seed: int) -> dict:
    """
    Run one full episode and return metrics dict.

    Returns
    -------
    dict with keys:
      total_reward, sla_violations, sla_rate,
      avg_latency, p95_latency, total_cost,
      n_scaling_events, vm_variance
    """
    env = CloudEnv(workload_mode=workload_mode, seed=seed)
    state = env.reset()
    policy.reset()

    total_reward    = 0.0
    sla_violations  = 0
    latencies       = []
    vm_counts       = []
    scaling_events  = 0
    prev_action     = DELTA_TO_ACTION[0]

    done = False
    while not done:
        action = policy.select_action(state)
        state, reward, done, info = env.step(action)

        total_reward   += reward
        sla_violations += info['sla_violation']
        latencies.append(info['latency'])
        vm_counts.append(info['n_active'])

        if ACTION_MAP[action] != 0:
            scaling_events += 1

    return {
        'total_reward'    : total_reward,
        'sla_violations'  : sla_violations,
        'sla_rate'        : sla_violations / len(latencies),
        'avg_latency'     : float(np.mean(latencies)),
        'p95_latency'     : float(np.percentile(latencies, 95)),
        'total_cost'      : float(np.sum(vm_counts)),
        'n_scaling_events': scaling_events,
        'vm_variance'     : float(np.var(vm_counts)),
    }


def evaluate_policy(policy, workload_mode: str, n_runs: int = 5, base_seed: int = 42):
    """
    Run policy n_runs times and report mean ± std for all metrics.
    Uses different seeds per run for statistical validity.
    """
    results = [run_episode(policy, workload_mode, seed=base_seed + i) for i in range(n_runs)]

    metrics = list(results[0].keys())
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results]
        summary[m] = {'mean': np.mean(vals), 'std': np.std(vals)}

    return summary


def print_summary(name: str, summary: dict):
    print(f"\n{'─'*55}")
    print(f"  Policy: {name}")
    print(f"{'─'*55}")
    print(f"  {'Metric':<25} {'Mean':>10}  {'±Std':>10}")
    print(f"  {'─'*45}")
    for metric, vals in summary.items():
        print(f"  {metric:<25} {vals['mean']:>10.3f}  {vals['std']:>10.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main  (run:  python baselines.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    N_RUNS = 5

    policies = [
        StaticPolicy(),
        ThresholdPolicy(),
        AutoScalePolicy(),
    ]

    workload_modes = ['sinusoidal', 'spike', 'poisson']

    print("\n" + "=" * 55)
    print("  BASELINE EVALUATION  (5 runs × 3 workload modes)")
    print("=" * 55)

    for mode in workload_modes:
        print(f"\n\n{'█'*55}")
        print(f"  WORKLOAD MODE: {mode.upper()}")
        print(f"{'█'*55}")

        for policy in policies:
            summary = evaluate_policy(policy, workload_mode=mode, n_runs=N_RUNS)
            print_summary(policy.name, summary)

    print(f"\n\n{'='*55}")
    print("  All baselines complete.")
    print(f"{'='*55}\n")