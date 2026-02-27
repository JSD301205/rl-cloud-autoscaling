"""
cloud_env.py — Cloud Autoscaling Environment (MDP Simulator)
Team Member 1 deliverable

Implements the CloudEnv that ALL teammates (DQN, PPO, baselines) will use.

Interface:
    env = CloudEnv(workload_mode='sinusoidal')
    state = env.reset()
    state, reward, done, info = env.step(action)

Action space : {0,1,2,3,4}  →  {-2,-1,0,+1,+2}  (index → delta)
State vector : [λ̃_t, λ̃_{t-1}, λ̃_{t-2}, u_t, n_t, q_t, b_t, a_{t-1}]  (8-dim)
"""

import numpy as np
from collections import deque

# Local import — workload.py must be in the same folder
from workload import WorkloadGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Environment Constants  (frozen — do not change)
# ──────────────────────────────────────────────────────────────────────────────
MAX_VMS          = 20       # hard ceiling on active VMs
MIN_VMS          = 1        # hard floor
BOOT_DELAY       = 3        # steps before a new VM becomes active
MONITORING_DELAY = 1        # steps of observation lag
EPISODE_LENGTH   = 500      # steps per episode
MU               = 20.0     # service rate per VM (req/step)
LATENCY_BASE     = 1.0      # L_base in latency formula
LATENCY_THRESH   = 5.0      # SLA violation threshold
VM_COST          = 1.0      # cost per active VM per step

# Reward weights
W_SLA   = 10.0   # SLA violation penalty
W_COST  =  1.0   # VM cost weight
W_ACT   =  0.5   # action magnitude penalty

# Discrete action map:  action_index → VM delta
ACTION_MAP = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
ACTION_SPACE_SIZE = len(ACTION_MAP)


# ──────────────────────────────────────────────────────────────────────────────
# CloudEnv
# ──────────────────────────────────────────────────────────────────────────────
class CloudEnv:
    """
    Simulates a single-tier cloud service with:
      - Boot delay for newly provisioned VMs
      - Monitoring delay (agent sees lagged arrival rates)
      - Queue accumulation under overload
      - SLA-based reward signal

    Parameters
    ----------
    workload_mode : str
        One of 'sinusoidal', 'spike', 'poisson'
    seed : int
        RNG seed for reproducibility
    """

    def __init__(self, workload_mode: str = 'sinusoidal', seed: int = 42):
        self.workload = WorkloadGenerator(mode=workload_mode, seed=seed)
        self.workload_mode = workload_mode
        self.seed = seed
        # State will be initialised in reset()
        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    # Gym-style interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset environment to initial state. Returns first state vector."""
        self.t = 0
        self.n_active   = 5           # start with 5 active VMs
        self.queue      = 0.0         # pending requests
        self.prev_action = 2          # index 2 → delta 0 (no-op)

        # Boot queue: list of remaining boot steps for each booting VM
        # e.g. [3, 2] means one VM with 3 steps left, one with 2 steps left
        self.boot_queue: list[int] = []

        # Arrival rate history for monitoring delay + state window
        # Pre-fill with zeros (episode hasn't started)
        self.arrival_history: deque = deque(
            [0.0] * (MONITORING_DELAY + 3), maxlen=MONITORING_DELAY + 3
        )

        # Reset workload generator
        self.workload.reset()

        # Collect first real arrival rate and build initial state
        lambda_t = self.workload.get(self.t)
        self.arrival_history.append(lambda_t)

        return self._get_state()

    def step(self, action: int):
        """
        Apply action, advance simulation by one step.

        Parameters
        ----------
        action : int
            Index in [0..4]  →  mapped to VM delta via ACTION_MAP

        Returns
        -------
        state  : np.ndarray  shape (8,)
        reward : float
        done   : bool
        info   : dict        (metrics for logging)
        """
        assert action in ACTION_MAP, f"Invalid action {action}. Must be in {list(ACTION_MAP.keys())}"

        # ── 1. Decode action ────────────────────────────────────────────
        delta = ACTION_MAP[action]

        # ── 2. Add VMs to boot queue (scale-up) ─────────────────────────
        vms_to_add = max(0, delta)
        for _ in range(vms_to_add):
            # Only add if we won't exceed MAX_VMS (counting booting too)
            projected = self.n_active + len(self.boot_queue) + 1
            if projected <= MAX_VMS:
                self.boot_queue.append(BOOT_DELAY)

        # ── 3. Scale down (immediate — decommission active VMs) ──────────
        vms_to_remove = max(0, -delta)
        if vms_to_remove > 0:
            self.n_active = max(MIN_VMS, self.n_active - vms_to_remove)

        # ── 4. Tick boot timers; graduate ready VMs ──────────────────────
        still_booting = []
        for timer in self.boot_queue:
            timer -= 1
            if timer <= 0:
                # VM is ready — add to active pool (respect MAX_VMS)
                if self.n_active < MAX_VMS:
                    self.n_active += 1
                # else silently drop (shouldn't happen with guard above)
            else:
                still_booting.append(timer)
        self.boot_queue = still_booting

        # ── 5. Get true arrival rate for this step ───────────────────────
        lambda_t = self.workload.get(self.t)
        self.arrival_history.append(lambda_t)    # history[-1] = newest

        # ── 6. Compute capacity & update queue ───────────────────────────
        capacity = self.n_active * MU
        self.queue = max(0.0, self.queue + lambda_t - capacity)

        # ── 7. Compute latency & SLA violation ───────────────────────────
        if capacity > 0:
            latency = LATENCY_BASE + self.queue / capacity
        else:
            latency = float('inf')

        sla_violation = 1 if latency > LATENCY_THRESH else 0

        # ── 8. Reward ────────────────────────────────────────────────────
        reward = (
            - W_SLA  * sla_violation
            - W_COST * self.n_active
            - W_ACT  * abs(delta)
        )

        # ── 9. Advance time ──────────────────────────────────────────────
        self.prev_action = action
        self.t += 1
        done = (self.t >= EPISODE_LENGTH)

        # ── 10. Build next state ─────────────────────────────────────────
        state = self._get_state()

        # ── 11. Info dict (for evaluation / logging) ─────────────────────
        info = {
            'step'          : self.t,
            'lambda_true'   : lambda_t,
            'lambda_obs'    : state[0],     # delayed observation
            'n_active'      : self.n_active,
            'n_booting'     : len(self.boot_queue),
            'queue'         : self.queue,
            'capacity'      : capacity,
            'latency'       : latency,
            'sla_violation' : sla_violation,
            'utilization'   : state[3],
            'reward'        : reward,
        }

        return state, reward, done, info

    # ──────────────────────────────────────────────────────────────────────
    # State construction
    # ──────────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Build 8-dim state vector:
        [λ̃_t, λ̃_{t-1}, λ̃_{t-2}, u_t, n_t, q_t, b_t, a_{t-1}]

        Monitoring delay: agent observes λ̃_t = λ_{t - MONITORING_DELAY}
        """
        history = list(self.arrival_history)   # oldest → newest

        # Delayed observations (monitoring delay applied)
        # history[-1] is the most recent true value
        # history[-(1+MONITORING_DELAY)] is what the agent can see "now"
        lam_obs_0 = history[-(1 + MONITORING_DELAY)]      # λ̃_t
        lam_obs_1 = history[-(2 + MONITORING_DELAY)]      # λ̃_{t-1}
        lam_obs_2 = history[-(3 + MONITORING_DELAY)]      # λ̃_{t-2}

        # CPU utilisation  u_t = λ_t / (n_t * μ)
        capacity = self.n_active * MU
        lambda_true = history[-1]
        utilization = lambda_true / capacity if capacity > 0 else 1.0
        utilization = min(utilization, 5.0)   # cap for stability

        state = np.array([
            lam_obs_0,              # 0: delayed arrival rate t
            lam_obs_1,              # 1: delayed arrival rate t-1
            lam_obs_2,              # 2: delayed arrival rate t-2
            utilization,            # 3: CPU utilisation
            float(self.n_active),   # 4: active VMs
            self.queue,             # 5: queue length
            float(len(self.boot_queue)),  # 6: booting VMs
            float(self.prev_action),      # 7: previous action index
        ], dtype=np.float32)

        return state

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return 8

    @property
    def action_dim(self) -> int:
        return ACTION_SPACE_SIZE

    def __repr__(self):
        return (
            f"CloudEnv(mode={self.workload_mode}, "
            f"t={self.t}, n={self.n_active}, q={self.queue:.1f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity test  (run:  python cloud_env.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("CloudEnv Sanity Test — No-op policy (action=2 always)")
    print("=" * 60)

    env = CloudEnv(workload_mode='sinusoidal', seed=42)
    state = env.reset()

    print(f"\nState dimensions : {env.state_dim}")
    print(f"Action dimensions: {env.action_dim}")
    print(f"Initial state    : {state}\n")

    total_reward   = 0.0
    sla_violations = 0
    latencies      = []

    done = False
    while not done:
        action = 2  # no-op
        state, reward, done, info = env.step(action)
        total_reward   += reward
        sla_violations += info['sla_violation']
        latencies.append(info['latency'])

    print(f"Episode complete ({env.t} steps)")
    print(f"  Total reward      : {total_reward:.2f}")
    print(f"  SLA violations    : {sla_violations} / {env.t}  ({100*sla_violations/env.t:.1f}%)")
    print(f"  Avg latency       : {np.mean(latencies):.3f}")
    print(f"  P95 latency       : {np.percentile(latencies, 95):.3f}")
    print(f"  Final state       : {state}")
    print("\nAll checks passed! CloudEnv is ready for your teammates.")
