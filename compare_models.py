import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import pandas as pd

# CHANGE THIS LINE:
# from dqn_agent import DQNAgent  (OLD - WRONG)
from agents.dqn_agent import DQNAgent  # NEW - CORRECT

from environment.traffic_env import TrafficLightEnv
# ... rest of your code

class ModelComparator:
    def __init__(self, n_episodes=50):
        self.n_episodes = n_episodes
        self.env = TrafficLightEnv()
        self.results = defaultdict(list)
        
    def random_policy(self):
        """Baseline 1: Random actions"""
        episode_metrics = []
        for ep in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.random.randint(0, 2)  # Random action
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
            episode_metrics.append({
                'reward': total_reward,
                'waiting_time': info['waiting_time'],
                'throughput': info['vehicles_passed']
            })
        return episode_metrics
    
    def fixed_timing_policy(self, green_duration=30):
        """Baseline 2: Traditional fixed-time traffic light"""
        episode_metrics = []
        for ep in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            while not done:
                # Switch every 'green_duration' steps
                action = 0 if (steps // green_duration) % 2 == 0 else 1
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
            episode_metrics.append({
                'reward': total_reward,
                'waiting_time': info['waiting_time'],
                'throughput': info['vehicles_passed']
            })
        return episode_metrics
    
    def rule_based_policy(self):
        """Baseline 3: Give green to longer queue"""
        episode_metrics = []
        for ep in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Rule: serve the direction with more vehicles
                action = 0 if state[0] > state[1] else 1  # Compare NS vs EW queue
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
            episode_metrics.append({
                'reward': total_reward,
                'waiting_time': info['waiting_time'],
                'throughput': info['vehicles_passed']
            })
        return episode_metrics
    
    def dqn_policy(self, model_path='models/best_model.pt'):
        """Your trained DQN model"""
        # Load your trained agent
        agent = DQNAgent(7, 2)
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval()
        
        episode_metrics = []
        for ep in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(state, epsilon=0.0)  # No exploration
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
            episode_metrics.append({
                'reward': total_reward,
                'waiting_time': info['waiting_time'],
                'throughput': info['vehicles_passed']
            })
        return episode_metrics
    
    def run_comparison(self):
        """Run all models and collect results"""
        print("🔄 Running Multi-Model Comparison...")
        print(f"   Evaluating each model over {self.n_episodes} episodes\n")
        
        # Run all policies
        print("📊 Random Policy (Baseline)...")
        self.results['Random'] = self.random_policy()
        
        print("📊 Fixed Timing Policy (Traditional)...")
        self.results['Fixed Timing'] = self.fixed_timing_policy()
        
        print("📊 Rule-Based Policy (Heuristic)...")
        self.results['Rule-Based'] = self.rule_based_policy()
        
        print("📊 DQN Model (Your Agent)...")
        self.results['DQN'] = self.dqn_policy()
        
        # Calculate statistics
        self.summary = {}
        for model_name, episodes in self.results.items():
            self.summary[model_name] = {
                'Waiting Time': np.mean([ep['waiting_time'] for ep in episodes]),
                'Waiting Time Std': np.std([ep['waiting_time'] for ep in episodes]),
                'Throughput': np.mean([ep['throughput'] for ep in episodes]),
                'Reward': np.mean([ep['reward'] for ep in episodes])
            }
        
        return self.summary
    
    def print_results(self):
        """Print comparison table"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON RESULTS")
        print("="*70)
        
        # Create DataFrame for nice display
        df = pd.DataFrame(self.summary).T
        print(df.round(2))
        print("="*70)
        
        # Calculate improvement
        baseline_wait = self.summary['Random']['Waiting Time']
        dqn_wait = self.summary['DQN']['Waiting Time']
        improvement = ((baseline_wait - dqn_wait) / baseline_wait) * 100
        
        print(f"\n🎯 DQN IMPROVEMENT: {improvement:.1f}% better waiting time than random")
        
    def plot_comparison(self, save_path='model_comparison.png'):
        """Create comparison bar chart"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        models = list(self.summary.keys())
        x_pos = np.arange(len(models))
        
        # Plot 1: Waiting Time
        waiting_means = [self.summary[m]['Waiting Time'] for m in models]
        waiting_stds = [self.summary[m]['Waiting Time Std'] for m in models]
        
        axes[0].bar(x_pos, waiting_means, yerr=waiting_stds, capsize=5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].set_ylabel('Waiting Time')
        axes[0].set_title('Average Waiting Time (Lower is Better)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput
        throughput_means = [self.summary[m]['Throughput'] for m in models]
        axes[1].bar(x_pos, throughput_means)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(models, rotation=45)
        axes[1].set_ylabel('Vehicles Passed')
        axes[1].set_title('Throughput (Higher is Better)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Reward
        reward_means = [self.summary[m]['Reward'] for m in models]
        axes[2].bar(x_pos, reward_means)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(models, rotation=45)
        axes[2].set_ylabel('Reward')
        axes[2].set_title('Average Reward')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Comparison chart saved to: {save_path}")
        plt.show()

if __name__ == "__main__":
    comparator = ModelComparator(n_episodes=50)
    comparator.run_comparison()
    comparator.print_results()
    comparator.plot_comparison()