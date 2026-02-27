"""
workload.py — Workload Generator for Cloud Autoscaling RL Project
Team Member 1 deliverable

Supports 3 modes:
  - sinusoidal   : smooth periodic traffic
  - spike        : sudden burst events
  - poisson      : Poisson-sampled bursty arrivals
"""

import numpy as np


class WorkloadGenerator:
    """
    Generates arrival rates (lambda_t) for each time step.

    Usage:
        wl = WorkloadGenerator(mode='sinusoidal', seed=42)
        lambda_t = wl.get(t)        # call each step
        wl.reset()                  # reset for new episode
    """

    MODES = ['sinusoidal', 'spike', 'poisson']

    def __init__(self, mode: str = 'sinusoidal', seed: int = 42):
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        self.mode = mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._spike_schedule = []   # pre-generated spike times
        self._poisson_intensity = 50.0  # current intensity level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, t: int) -> float:
        """Return arrival rate at time step t."""
        if self.mode == 'sinusoidal':
            return self._sinusoidal(t)
        elif self.mode == 'spike':
            return self._spike(t)
        elif self.mode == 'poisson':
            return self._poisson(t)

    def reset(self):
        """Reset internal state for a fresh episode."""
        self.rng = np.random.default_rng(self.seed)
        self._spike_schedule = []
        self._poisson_intensity = 50.0

    # ------------------------------------------------------------------
    # Workload modes
    # ------------------------------------------------------------------

    def _sinusoidal(self, t: int) -> float:
        """
        Smooth sinusoidal load.
        Base load oscillates between ~20 and ~100 req/step.
        Small Gaussian noise added for realism.
        """
        base = 60.0
        amplitude = 40.0
        period = 100          # steps per full cycle
        noise = self.rng.normal(0, 3)
        rate = base + amplitude * np.sin(2 * np.pi * t / period) + noise
        return max(0.0, rate)

    def _spike(self, t: int) -> float:
        """
        Normal background load with sudden burst spikes.
        Spikes are pre-scheduled at random intervals of 50-150 steps,
        lasting 5-15 steps each.
        """
        # Generate spike schedule lazily on first call
        if not self._spike_schedule:
            self._spike_schedule = self._generate_spike_schedule(episode_len=500)

        base = 40.0 + self.rng.normal(0, 3)
        # Check if current t falls inside a spike window
        for (start, end, magnitude) in self._spike_schedule:
            if start <= t < end:
                return max(0.0, base + magnitude)
        return max(0.0, base)

    def _poisson(self, t: int) -> float:
        """
        Poisson-sampled arrivals with randomly shifting intensity.
        Intensity drifts slowly; occasional jumps simulate bursty traffic.
        """
        # Occasionally shift intensity level
        if self.rng.random() < 0.05:   # 5% chance each step
            jump = self.rng.choice([-20, -10, 10, 20, 30])
            self._poisson_intensity = float(
                np.clip(self._poisson_intensity + jump, 10, 120)
            )

        # Poisson sample around current intensity
        rate = self.rng.poisson(lam=self._poisson_intensity)
        return float(rate)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _generate_spike_schedule(self, episode_len: int):
        """Pre-generate (start, end, magnitude) tuples for spike mode."""
        schedule = []
        t = 0
        while t < episode_len:
            gap = int(self.rng.integers(50, 150))
            t += gap
            if t >= episode_len:
                break
            duration = int(self.rng.integers(5, 15))
            magnitude = float(self.rng.uniform(80, 160))  # big spike
            schedule.append((t, t + duration, magnitude))
            t += duration
        return schedule


# ------------------------------------------------------------------
# Quick sanity test  (run:  python workload.py)
# ------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = 500
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Workload Generator — All 3 Modes', fontsize=14)

    for ax, mode in zip(axes, WorkloadGenerator.MODES):
        wl = WorkloadGenerator(mode=mode, seed=42)
        rates = [wl.get(t) for t in range(steps)]
        ax.plot(rates, linewidth=1.2)
        ax.set_title(f'Mode: {mode}')
        ax.set_ylabel('Arrival Rate (req/step)')
        ax.set_xlabel('Time Step')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('workload_preview.png', dpi=120)
    print("Saved workload_preview.png")
    print("All 3 modes working correctly!")
