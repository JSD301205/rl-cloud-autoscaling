"""
ppo_agent.py — Proximal Policy Optimization for Cloud Autoscaling
Team Member 3 deliverable

Implements PPO with:
  - Actor-Critic architecture (separate policy & value networks)
  - Clipped surrogate objective with Exact Gradient Masking
  - Generalized Advantage Estimation (GAE)
  - Mini-batch updates with early stopping (KL Divergence)
  - Global Gradient Clipping
  - Vectorized operations

Interface:
    agent = PPOAgent(state_dim=8, action_dim=5)
    agent.train(env, n_episodes=1000)
    action, _, _ = agent.select_action(state, deterministic=True)
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
    
    def __init__(self, in_features: int, out_features: int, gain: float = np.sqrt(2.0)):
        self.W = np.random.randn(in_features, out_features) * (gain / np.sqrt(in_features))
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


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    shifted = np.clip(shifted, -50.0, 50.0)
    exp_x = np.exp(shifted)
    probs = exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sum = np.sum(probs, axis=-1, keepdims=True)
    probs = probs / (probs_sum + 1e-12)
    return probs


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
# Actor Network (Policy)
# ─────────────────────────────────────────────────────────────────────────────
class ActorNetwork:
    """
    Policy network: state → action probabilities
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4):
        self.fc1 = Linear(state_dim, 64)
        self.fc2 = Linear(64, 64)
        # FIX: Near-zero initialization for the final layer to ensure uniform starting probabilities
        self.fc3 = Linear(64, action_dim, gain=0.01)
        self.lr = lr
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        self.z1 = self.fc1.forward(state)
        self.a1 = relu(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        self.a2 = relu(self.z2)
        self.logits = self.fc3.forward(self.a2)
        self.probs = softmax(self.logits)
        return self.probs
    
    def backward(self, grad_logits: np.ndarray, max_grad_norm: float):
        """Backprop through network with FIX 1: Proper Gradient Clipping."""
        # Calculate gradients (don't update weights yet)
        grad_a2 = self.fc3.backward(grad_logits)
        grad_z2 = grad_a2 * relu_grad(self.z2)
        grad_a1 = self.fc2.backward(grad_z2)
        grad_z1 = grad_a1 * relu_grad(self.z1)
        self.fc1.backward(grad_z1)
        
        # FIX 1: Compute global norm
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
                
        # Now apply the learning rate update
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


# ─────────────────────────────────────────────────────────────────────────────
# Critic Network (Value Function)
# ─────────────────────────────────────────────────────────────────────────────
class CriticNetwork:
    """
    Value network: state → V(s)
    """
    def __init__(self, state_dim: int, lr: float = 1e-3):
        self.fc1 = Linear(state_dim, 64)
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, 1)
        self.lr = lr
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        self.z1 = self.fc1.forward(state)
        self.a1 = relu(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        self.a2 = relu(self.z2)
        self.value = self.fc3.forward(self.a2)
        return self.value.squeeze()
    
    def backward(self, grad_value: np.ndarray, max_grad_norm: float):
        """Backprop through network with FIX 1: Proper Gradient Clipping."""
        # Calculate gradients
        grad_a2 = self.fc3.backward(grad_value)
        grad_z2 = grad_a2 * relu_grad(self.z2)
        grad_a1 = self.fc2.backward(grad_z2)
        grad_z1 = grad_a1 * relu_grad(self.z1)
        self.fc1.backward(grad_z1)
        
        # FIX 1: Compute global norm
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
                
        # Now apply the learning rate update
        for layer in [self.fc1, self.fc2, self.fc3]:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()
        
    def store(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.dones, self.values, self.log_probs = [], [], []
    
    def get(self):
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
        }
    
    def __len__(self):
        return len(self.states)


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = ACTION_SPACE_SIZE,
        lr_actor: float = 5e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,  # FIX 5: Lowered to 0.01 for proper convergence
        target_kl: float = 0.015,    # FIX 4: Target KL for early stopping
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(state_dim, action_dim, lr_actor)
        self.critic = CriticNetwork(state_dim, lr_critic)
        self.buffer = RolloutBuffer()
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.state_norm = RunningNorm(state_dim)

    def _normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        if update_stats:
            self.state_norm.update(state)
        return self.state_norm.normalize(state)

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Pure PPO Action Selection (No Epsilon-Greedy)."""
        state = self._normalize_state(state, update_stats=not deterministic)
        state = state.reshape(1, -1)
        probs = self.actor.forward(state).squeeze()
        
        # Sanitize probabilities
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs)
        
        value = self.critic.forward(state)
        
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.action_dim, p=probs))
            
        log_prob = np.log(np.clip(probs[action], 1e-8, 1.0))
        return action, value, log_prob
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards)
        gae = 0.0
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def update(self):
        data = self.buffer.get()
        states, actions, old_log_probs = data['states'], data['actions'], data['log_probs']
        values, rewards, dones = data['values'], data['rewards'], data['dones']
        
        next_value = 0.0 if dones[-1] else self.critic.forward(states[-1:])
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            approx_kl_epoch = []
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                b_size = len(b_actions)
                
                # ── Actor Update ──────────────────────────────────────
                probs = self.actor.forward(b_states)
                probs = np.clip(probs, 1e-8, 1.0)
                probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-12)
                
                action_probs = probs[np.arange(b_size), b_actions]
                log_probs = np.log(action_probs + 1e-8)
                
                # FIX 4: Track KL Divergence
                log_ratio = log_probs - b_old_log_probs
                kl = np.mean((np.exp(log_ratio) - 1) - log_ratio)
                approx_kl_epoch.append(kl)
                
                ratio = np.exp(log_ratio)
                
                # FIX 3: Correct Clipped Gradient Logic
                # We want to minimize the negative surrogate objective
                # Derivative is -ratio * advantage, UNLESS clipping active
                grad_log_probs = -ratio * b_advantages / b_size
                
                # Mask out gradients where clipping condition is met
                is_clipped = np.logical_or(
                    np.logical_and(b_advantages > 0, ratio > 1 + self.clip_epsilon),
                    np.logical_and(b_advantages < 0, ratio < 1 - self.clip_epsilon)
                )
                grad_log_probs[is_clipped] = 0.0
                
                # Gradient w.r.t. probs
                grad_probs = np.zeros_like(probs)
                grad_probs[np.arange(b_size), b_actions] = grad_log_probs / (action_probs + 1e-8)
                
                # Entropy gradient (minimizing negative entropy)
                grad_entropy = self.entropy_coef * (np.log(probs + 1e-8) + 1) / b_size
                grad_probs -= grad_entropy
                
                # Backprop to logits
                grad_logits = probs * (grad_probs - np.sum(grad_probs * probs, axis=1, keepdims=True))
                
                # Update Actor (Passes max_grad_norm for FIX 1)
                self.actor.backward(grad_logits, self.max_grad_norm)
                
                # ── Critic Update ─────────────────────────────────────
                # FIX 2: Vectorized Critic Update
                predicted_values = self.critic.forward(b_states)
                if predicted_values.ndim == 1:
                    predicted_values = predicted_values.reshape(-1, 1)
                
                b_returns_reshaped = b_returns.reshape(-1, 1)
                
                # MSE Gradient: d( (v - ret)^2 )/dv = 2*(v - ret)
                grad_values = 2 * (predicted_values - b_returns_reshaped) / b_size
                
                # Update Critic (Passes max_grad_norm for FIX 1)
                self.critic.backward(grad_values, self.max_grad_norm)
            
            # FIX 4: Early Stopping based on KL Divergence
            avg_kl = np.mean(approx_kl_epoch)
            if self.target_kl is not None and avg_kl > 1.5 * self.target_kl:
                # print(f"  [Epoch {epoch}] Early stopping (KL: {avg_kl:.4f} > Target: {self.target_kl})")
                break
                
        self.buffer.clear()
    
    def train(self, env, n_episodes=1000, eval_freq=50, save_path='ppo_agent.pkl', verbose=True):
        episode_rewards_window = deque(maxlen=100)
        for episode in range(1, n_episodes + 1):
            state = env.reset()
            episode_reward, episode_length = 0.0, 0
            done = False
            
            while not done:
                action, value, log_prob = self.select_action(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                
                # FIX: Scale reward to prevent Exploding Gradients in the Critic Network
                scaled_reward = reward * 0.01
                
                norm_state = self._normalize_state(state, update_stats=False)
                self.buffer.store(norm_state, action, scaled_reward, done, value, log_prob)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            if len(self.buffer) > 0:
                self.update()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode_rewards_window.append(episode_reward)
            
            if verbose and episode % eval_freq == 0:
                print(f"Episode {episode}/{n_episodes} | Avg Reward (100): {np.mean(episode_rewards_window):.2f}")
        
        self.save(save_path)
    
    def save(self, path: str):
        data = {
            'actor_fc1_W': self.actor.fc1.W, 'actor_fc1_b': self.actor.fc1.b,
            'actor_fc2_W': self.actor.fc2.W, 'actor_fc2_b': self.actor.fc2.b,
            'actor_fc3_W': self.actor.fc3.W, 'actor_fc3_b': self.actor.fc3.b,
            'critic_fc1_W': self.critic.fc1.W, 'critic_fc1_b': self.critic.fc1.b,
            'critic_fc2_W': self.critic.fc2.W, 'critic_fc2_b': self.critic.fc2.b,
            'critic_fc3_W': self.critic.fc3.W, 'critic_fc3_b': self.critic.fc3.b,
            'state_norm_mean': self.state_norm.mean,
            'state_norm_m2': self.state_norm.m2,
            'state_norm_count': self.state_norm.count,
        }
        with open(path, 'wb') as f: pickle.dump(data, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f: data = pickle.load(f)
        self.actor.fc1.W, self.actor.fc1.b = data['actor_fc1_W'], data['actor_fc1_b']
        self.actor.fc2.W, self.actor.fc2.b = data['actor_fc2_W'], data['actor_fc2_b']
        self.actor.fc3.W, self.actor.fc3.b = data['actor_fc3_W'], data['actor_fc3_b']
        self.critic.fc1.W, self.critic.fc1.b = data['critic_fc1_W'], data['critic_fc1_b']
        self.critic.fc2.W, self.critic.fc2.b = data['critic_fc2_W'], data['critic_fc2_b']
        self.critic.fc3.W, self.critic.fc3.b = data['critic_fc3_W'], data['critic_fc3_b']
        self.state_norm.mean = data['state_norm_mean']
        self.state_norm.m2 = data['state_norm_m2']
        self.state_norm.count = data['state_norm_count']

if __name__ == '__main__':
    print("  Training Advanced PPO Agent...")
    env = CloudEnv(workload_mode='sinusoidal', seed=42)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.train(env, n_episodes=1500, save_path='ppo_agent.pkl')