# agents/dqn_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    """Deep Q-Network - Slightly wider for better learning"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """Experience Replay Buffer - INCREASED CAPACITY"""
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for Traffic Light Control - WITH BEST MODEL SAVING"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # Default hyperparameters - FINAL OPTIMIZED
        self.config = config or {
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.997,        # CHANGED: from 0.999 to 0.997 (optimal)
            'batch_size': 64,
            'memory_size': 20000,
            'target_update': 50
        }
        
        # Add device to config if not present
        if 'device' not in self.config:
            self.config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Networks
        self.device = torch.device(self.config['device'])
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=self.config['learning_rate'])
        
        # Experience replay
        self.memory = ReplayBuffer(self.config['memory_size'])
        
        # Training variables
        self.epsilon = self.config['epsilon']
        self.steps_done = 0
        # Track best performance
        self.best_reward = -float('inf')
        self.best_model_path = '../models/best_model.pt'
        
        # Initialize target network
        self.update_target_network()
    
    def act(self, state, eval_mode=False):
        """Choose action using epsilon-greedy policy with exploration noise"""
        # During training: use epsilon-greedy
        if not eval_mode and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # During evaluation: add small exploration noise to prevent getting stuck
        if eval_mode and random.random() < 0.05:  # 5% random actions during evaluation
            return random.randrange(self.action_size)
        
        # Normal action selection
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self):
        """Update policy network using batch of experiences"""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config['batch_size']
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values using TARGET NETWORK
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.config['gamma'] * next_q * (1 - dones))
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Gradient clipping (prevents huge updates)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.config['epsilon_min'], 
                          self.epsilon * self.config['epsilon_decay'])
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_best_if_needed(self, current_reward, episode):
        """Save model if it's the best so far"""
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.save(self.best_model_path)
            print(f"\n🏆 NEW BEST MODEL! Episode {episode}, Reward: {current_reward:.2f}")
            return True
        return False
    
    def save(self, filepath):
        """Save agent"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'best_reward': self.best_reward
        }, filepath)
    
    def load(self, filepath):
        """Load agent"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.best_reward = checkpoint.get('best_reward', -float('inf'))