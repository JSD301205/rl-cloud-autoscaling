"""
q_learning_agent.py — Tabular Q-Learning for Cloud Autoscaling
Team Member 3 deliverable (Foundational Q-Learning)

Implements classic Q-Learning with:
  - State space discretization (binning)
  - Q-table (dictionary-based)
  - Epsilon-greedy exploration
  - Learning rate & discount factor
  - State normalization for consistent discretization

Interface:
    agent = QLearningAgent(state_dim=8, action_dim=5, n_bins=10)
    agent.train(env, n_episodes=1000)
    action = agent.select_action(state, deterministic=True)

Note: This is a foundational approach. For continuous state spaces,
      Deep Q-Learning (DQN) is more practical.
"""

import numpy as np
import sys
import os
from collections import defaultdict, deque
import pickle

# Path fix for CloudEnv import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
from cloud_env import CloudEnv, ACTION_SPACE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# State Preprocessor (Discretization)
# ─────────────────────────────────────────────────────────────────────────────
class StateDiscretizer:
    """
    Converts continuous states into discrete buckets for tabular Q-learning.
    
    Uses running statistics to adapt bucket boundaries online.
    """
    
    def __init__(self, state_dim: int, n_bins: int = 10, clip: float = 3.0):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of state space
        n_bins : int
            Number of bins per dimension (total states = n_bins^state_dim)
        clip : float
            Clip normalized values to [-clip, clip]
        """
        self.state_dim = state_dim
        self.n_bins = n_bins
        self.clip = clip
        
        # Running statistics
        self.count = 0
        self.mean = np.zeros(state_dim, dtype=np.float64)
        self.m2 = np.zeros(state_dim, dtype=np.float64)
    
    def update(self, state: np.ndarray):
        """Update statistics with new state."""
        state = np.asarray(state, dtype=np.float64)
        self.count += 1
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.m2 += delta * delta2
    
    @property
    def std(self) -> np.ndarray:
        """Compute standard deviation from running variance."""
        if self.count < 2:
            return np.ones(self.state_dim, dtype=np.float64)
        var = self.m2 / (self.count - 1)
        return np.sqrt(np.maximum(var, 1e-8))
    
    def discretize(self, state: np.ndarray, update_stats: bool = True) -> tuple:
        """
        Discretize state into bin indices.
        
        Parameters
        ----------
        state : np.ndarray
            Raw state vector
        update_stats : bool
            Whether to update running statistics
        
        Returns
        -------
        tuple
            Bin indices, one per state dimension. Can be used as dict key.
        """
        if update_stats:
            self.update(state)
        
        # Normalize
        state = np.asarray(state, dtype=np.float64)
        z = (state - self.mean) / (self.std + 1e-8)
        z = np.clip(z, -self.clip, self.clip)
        
        # Map to [0, 1]
        z_scaled = (z + self.clip) / (2 * self.clip)
        
        # Bin
        bins = np.floor(z_scaled * self.n_bins).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        
        return tuple(bins)


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning Agent (Tabular)
# ─────────────────────────────────────────────────────────────────────────────
class QLearningAgent:
    """
    Classic Q-Learning agent using tabular approach.
    
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = ACTION_SPACE_SIZE,
        n_bins: int = 10,
        alpha: float = 0.1,          # Learning rate
        gamma: float = 0.99,         # Discount factor
        epsilon_start: float = 1.0,  # Exploration rate
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of state space
        action_dim : int
            Number of discrete actions
        n_bins : int
            Number of discretization bins per state dimension
        alpha : float
            Learning rate (0 < α ≤ 1)
        gamma : float
            Discount factor (0 ≤ γ ≤ 1)
        epsilon_start : float
            Initial exploration rate
        epsilon_end : float
            Minimum exploration rate
        epsilon_decay : float
            Multiplicative decay per episode
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State discretizer
        self.discretizer = StateDiscretizer(state_dim, n_bins)
        
        # Q-table: {(state_bins): [q_a0, q_a1, ...]}
        self.q_table = defaultdict(lambda: np.zeros(action_dim, dtype=np.float32))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_table_size_history = []
        self.alpha_decay = 0.9995  # Optional: gradual learning rate decay
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state (continuous)
        deterministic : bool
            If True, always take greedy action
        
        Returns
        -------
        int
            Selected action (0 to action_dim-1)
        """
        state_bins = self.discretizer.discretize(state, update_stats=not deterministic)
        
        # Epsilon-greedy
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        
        # Greedy: select Q-value maximizing action
        q_values = self.q_table[state_bins]
        return int(np.argmax(q_values))
    
    def update_q_value(self, state: np.ndarray, action: int, 
                      reward: float, next_state: np.ndarray, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') - Q(s,a) ]
        
        Parameters
        ----------
        state : np.ndarray
            Previous state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Resulting state
        done : bool
            Whether episode is finished
        """
        state_bins = self.discretizer.discretize(state, update_stats=False)
        next_state_bins = self.discretizer.discretize(next_state, update_stats=False)
        
        # Current Q-value
        q_current = self.q_table[state_bins][action]
        
        # Maximum Q-value of next state (exploitation target)
        if done:
            q_next_max = 0.0
        else:
            q_next_max = np.max(self.q_table[next_state_bins])
        
        # Q-learning update
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_current
        q_new = q_current + self.alpha * td_error
        
        self.q_table[state_bins][action] = q_new
    
    def train(self, env, n_episodes: int = 1000, eval_freq: int = 50, 
              save_path: str = 'q_learning_agent.pkl', verbose: bool = True):
        """
        Train Q-Learning agent.
        
        Parameters
        ----------
        env : CloudEnv
            Environment to train on
        n_episodes : int
            Number of training episodes
        eval_freq : int
            Evaluation frequency (episodes)
        save_path : str
            Path to save trained agent
        verbose : bool
            Print training progress
        """
        episode_rewards_window = deque(maxlen=100)
        
        for episode in range(1, n_episodes + 1):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Select and execute action
                action = self.select_action(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Optional: Decay learning rate
            # self.alpha *= self.alpha_decay
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode_rewards_window.append(episode_reward)
            self.q_table_size_history.append(len(self.q_table))
            
            if verbose and episode % eval_freq == 0:
                avg_reward = np.mean(episode_rewards_window)
                n_states = len(self.q_table)
                print(f"Episode {episode}/{n_episodes} | Avg Reward (100): {avg_reward:>7.2f} | "
                      f"States explored: {n_states:>5} | ε: {self.epsilon:.3f}")
        
        self.save(save_path)
        if verbose:
            print(f"\n✓ Training complete. Agent saved to {save_path}")
    
    def evaluate(self, env, n_episodes: int = 5) -> dict:
        """
        Evaluate agent on environment.
        
        Parameters
        ----------
        env : CloudEnv
            Environment to evaluate on
        n_episodes : int
            Number of evaluation episodes
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        rewards = []
        lengths = []
        
        for _ in range(n_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.select_action(state, deterministic=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
        }
    
    def save(self, path: str):
        """Save Q-table and discretizer to disk."""
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        
        data = {
            'q_table': q_table_dict,
            'discretizer_mean': self.discretizer.mean,
            'discretizer_m2': self.discretizer.m2,
            'discretizer_count': self.discretizer.count,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load Q-table and discretizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_dim, dtype=np.float32))
        for state_bins, q_values in data['q_table'].items():
            self.q_table[state_bins] = np.array(q_values, dtype=np.float32)
        
        self.discretizer.mean = data['discretizer_mean']
        self.discretizer.m2 = data['discretizer_m2']
        self.discretizer.count = data['discretizer_count']
        self.epsilon = data['epsilon']
        self.alpha = data['alpha']


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Training Tabular Q-Learning Agent")
    print("="*70 + "\n")
    
    env = CloudEnv(workload_mode='sinusoidal', seed=42)
    agent = QLearningAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_bins=10,          # Discretization
        alpha=0.1,          # Learning rate
        gamma=0.99,         # Discount factor
        epsilon_start=1.0,  # Full exploration at start
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    
    agent.train(env, n_episodes=1000, eval_freq=100, 
                save_path='q_learning_agent.pkl', verbose=True)
    
    print("\n" + "="*70)
    print("  Evaluation")
    print("="*70)
    metrics = agent.evaluate(env, n_episodes=5)
    print(f"  Avg Reward: {metrics['avg_reward']:.2f} (±{metrics['std_reward']:.2f})")
    print(f"  Avg Episode Length: {metrics['avg_length']:.1f}")
    print(f"  Total unique states explored: {len(agent.q_table)}")
    print("="*70 + "\n")
