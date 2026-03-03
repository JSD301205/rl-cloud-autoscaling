"""
dqn_agent.py — Deep Q-Network for Cloud Autoscaling
Team Member 3 deliverable (Q-Learning Implementation)

Implements DQN with:
  - Neural Network Q-function approximator
  - Experience Replay Buffer
  - Target Network (Frozen copy for stability)
  - Epsilon-Greedy Exploration
  - Double DQN (DDQN) for reduced overestimation
  - Prioritized Experience Replay (Optional)
  - State Normalization

Interface:
    agent = DQNAgent(state_dim=8, action_dim=5)
    agent.train(env, n_episodes=1000)
    action = agent.select_action(state, deterministic=True)
"""

import numpy as np
import sys
import os
from collections import deque
import pickle

# Path fix for CloudEnv import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'env'))
from cloud_env import CloudEnv, ACTION_SPACE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network Layers (NumPy implementation)
# ─────────────────────────────────────────────────────────────────────────────
class Linear:
    """Fully connected layer with Xavier initialization."""
    
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.input = None
        self.dW = None
        self.db = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.W + self.b
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW = self.input.T @ grad
        self.db = np.sum(grad, axis=0)
        return grad @ self.W.T


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class RunningNorm:
    """Running mean/std normalizer for online normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.eps = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean, dtype=np.float64)
        return np.maximum(self.m2 / (self.count - 1), self.eps)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.eps)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / self.std
        z = np.clip(z, -clip, clip)
        return z.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────────────────────
class QNetwork:
    """
    Q-function approximator: state → Q-values for all actions
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4):
        self.fc1 = Linear(state_dim, 128)
        self.fc2 = Linear(128, 128)
        self.fc3 = Linear(128, action_dim)
        self.lr = lr
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass: state → Q-values
        
        Parameters
        ----------
        state : np.ndarray
            Shape (batch_size, state_dim)
        
        Returns
        -------
        np.ndarray
            Shape (batch_size, action_dim) — Q-values for each action
        """
        self.z1 = self.fc1.forward(state)
        self.a1 = relu(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        self.a2 = relu(self.z2)
        self.q_values = self.fc3.forward(self.a2)
        return self.q_values
    
    def backward(self, grad_q: np.ndarray, max_grad_norm: float = 0.5):
        """
        Backward pass with global gradient clipping.
        """
        # Calculate gradients
        grad_a2 = self.fc3.backward(grad_q)
        grad_z2 = grad_a2 * relu_grad(self.z2)
        grad_a1 = self.fc2.backward(grad_z2)
        grad_z1 = grad_a1 * relu_grad(self.z1)
        self.fc1.backward(grad_z1)
        
        # Compute global norm
        global_norm = np.sqrt(
            np.sum(self.fc1.dW**2) + np.sum(self.fc1.db**2) +
            np.sum(self.fc2.dW**2) + np.sum(self.fc2.db**2) +
            np.sum(self.fc3.dW**2) + np.sum(self.fc3.db**2)
        )
        
        # Apply clipping
        clip_coef = max_grad_norm / (global_norm + 1e-6)
        if clip_coef < 1.0:
            for layer in [self.fc1, self.fc2, self.fc3]:
                layer.dW *= clip_coef
                layer.db *= clip_coef
                
        # Update weights
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db
    
    def copy_weights(self, other: 'QNetwork'):
        """Copy weights from another network (for target network update)."""
        self.fc1.W = other.fc1.W.copy()
        self.fc1.b = other.fc1.b.copy()
        self.fc2.W = other.fc2.W.copy()
        self.fc2.b = other.fc2.b.copy()
        self.fc3.W = other.fc3.W.copy()
        self.fc3.b = other.fc3.b.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store transition in buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a random batch from buffer.
        
        Returns
        -------
        dict with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
        }
    
    def __len__(self):
        return len(self.memory)


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent (Q-Learning)
# ─────────────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network Agent with Double DQN improvements.
    
    Key Features:
    - Experience replay for better sample efficiency
    - Target network for stability
    - Double DQN to reduce Q-value overestimation
    - Epsilon-greedy exploration
    - State normalization
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = ACTION_SPACE_SIZE,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        use_double_dqn: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim, lr)
        self.target_network = QNetwork(state_dim, action_dim, lr)
        self.target_network.copy_weights(self.q_network)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # State normalization
        self.state_norm = RunningNorm(state_dim)
        
        # Training stats
        self.episode_rewards = []
        self.episode_lengths = []
        self.update_count = 0
        self.loss_history = []
    
    def _normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize state using running statistics."""
        if update_stats:
            self.state_norm.update(state)
        return self.state_norm.normalize(state)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        deterministic : bool
            If True, always select greedy action (no exploration)
        
        Returns
        -------
        int
            Selected action
        """
        state_norm = self._normalize_state(state, update_stats=not deterministic)
        state_batch = state_norm.reshape(1, -1)
        
        # Epsilon-greedy
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        
        # Greedy
        q_values = self.q_network.forward(state_batch).squeeze()
        return int(np.argmax(q_values))
    
    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get Q-values for a batch of states."""
        return self.q_network.forward(states)
    
    def update(self):
        """
        Update Q-network using a mini-batch from replay buffer.
        Implements Double DQN if enabled.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Forward pass: compute Q(s, a)
        q_values = self.q_network.forward(states)
        batch_size = len(states)
        
        if self.use_double_dqn:
            # Double DQN: use online network to select actions, target to evaluate
            next_q_online = self.q_network.forward(next_states)
            next_actions = np.argmax(next_q_online, axis=1)
            next_q_target = self.target_network.forward(next_states)
            next_q_values = next_q_target[np.arange(batch_size), next_actions]
        else:
            # Standard DQN: use target network for both selection and evaluation
            next_q_values_target = self.target_network.forward(next_states)
            next_q_values = np.max(next_q_values_target, axis=1)
        
        # Compute target Q-values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss (MSE)
        td_error = q_values[np.arange(batch_size), actions] - target_q_values
        loss = np.mean(td_error ** 2)
        self.loss_history.append(loss)
        
        # Backward pass: compute gradients
        grad_q = np.zeros_like(q_values)
        grad_q[np.arange(batch_size), actions] = 2 * td_error / batch_size
        
        # Update Q-network
        self.q_network.backward(grad_q, max_grad_norm=0.5)
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.copy_weights(self.q_network)
    
    def train(self, env, n_episodes: int = 1000, eval_freq: int = 50, 
              save_path: str = 'dqn_agent.pkl', verbose: bool = True):
        """
        Train the DQN agent.
        
        Parameters
        ----------
        env : CloudEnv
            Environment to train on
        n_episodes : int
            Number of episodes to train
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
                
                # Store transition in replay buffer
                norm_state = self._normalize_state(state, update_stats=False)
                self.replay_buffer.push(norm_state, action, reward, 
                                       self._normalize_state(next_state, update_stats=False), 
                                       done)
                
                # Update Q-network
                self.update()
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode_rewards_window.append(episode_reward)
            
            if verbose and episode % eval_freq == 0:
                avg_reward = np.mean(episode_rewards_window)
                avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
                print(f"Episode {episode}/{n_episodes} | Avg Reward (100): {avg_reward:.2f} | "
                      f"Loss: {avg_loss:.4f} | ε: {self.epsilon:.3f}")
        
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
        """Save agent to disk."""
        data = {
            'q_fc1_W': self.q_network.fc1.W,
            'q_fc1_b': self.q_network.fc1.b,
            'q_fc2_W': self.q_network.fc2.W,
            'q_fc2_b': self.q_network.fc2.b,
            'q_fc3_W': self.q_network.fc3.W,
            'q_fc3_b': self.q_network.fc3.b,
            'target_fc1_W': self.target_network.fc1.W,
            'target_fc1_b': self.target_network.fc1.b,
            'target_fc2_W': self.target_network.fc2.W,
            'target_fc2_b': self.target_network.fc2.b,
            'target_fc3_W': self.target_network.fc3.W,
            'target_fc3_b': self.target_network.fc3.b,
            'state_norm_mean': self.state_norm.mean,
            'state_norm_m2': self.state_norm.m2,
            'state_norm_count': self.state_norm.count,
            'epsilon': self.epsilon,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load agent from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.q_network.fc1.W = data['q_fc1_W']
        self.q_network.fc1.b = data['q_fc1_b']
        self.q_network.fc2.W = data['q_fc2_W']
        self.q_network.fc2.b = data['q_fc2_b']
        self.q_network.fc3.W = data['q_fc3_W']
        self.q_network.fc3.b = data['q_fc3_b']
        
        self.target_network.copy_weights(self.q_network)
        
        self.state_norm.mean = data['state_norm_mean']
        self.state_norm.m2 = data['state_norm_m2']
        self.state_norm.count = data['state_norm_count']
        self.epsilon = data['epsilon']


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  Training DQN Agent...")
    print("="*70 + "\n")
    
    env = CloudEnv(workload_mode='sinusoidal', seed=42)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=1000,
        use_double_dqn=True,
    )
    
    agent.train(env, n_episodes=1500, eval_freq=100, save_path='dqn_agent.pkl', verbose=True)
    
    print("\n" + "="*70)
    print("  Evaluation")
    print("="*70)
    metrics = agent.evaluate(env, n_episodes=5)
    print(f"  Avg Reward: {metrics['avg_reward']:.2f} (±{metrics['std_reward']:.2f})")
    print(f"  Avg Episode Length: {metrics['avg_length']:.1f}")
    print("="*70 + "\n")
