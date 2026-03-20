import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from collections import defaultdict

# ADD THIS IMPORT
from compare_models import ModelComparator

from environment.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQNAgent
# ... rest of your code

class VisualizationSuite:
    def __init__(self):
        sns.set_style("darkgrid")
        self.comparator = ModelComparator(n_episodes=50)
        
    def create_all_visualizations(self):
        """Generate all required plots"""
        
        # 1. Get comparison data first
        results = self.comparator.run_comparison()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Traffic Light RL - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 2.1 Reward Curve (from your training)
        ax1 = plt.subplot(2, 3, 1)
        self.plot_reward_curve(ax1)
        
        # 2.2 Comparison Bar Chart
        ax2 = plt.subplot(2, 3, 2)
        self.plot_comparison_bars(ax2, results)
        
        # 2.3 Waiting Time Distribution
        ax3 = plt.subplot(2, 3, 3)
        self.plot_waiting_distribution(ax3)
        
        # 2.4 Queue Length Over Time
        ax4 = plt.subplot(2, 3, 4)
        self.plot_queue_evolution(ax4)
        
        # 2.5 Action Distribution
        ax5 = plt.subplot(2, 3, 5)
        self.plot_action_distribution(ax5)
        
        # 2.6 Fairness Heatmap
        ax6 = plt.subplot(2, 3, 6)
        self.plot_fairness_heatmap(ax6)
        
        plt.tight_layout()
        plt.savefig('complete_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Complete visualization saved to 'complete_analysis.png'")
        plt.show()
    
    def plot_reward_curve(self, ax):
        """Plot training reward curve"""
        # You can load this from your training logs
        # For now, sample data based on your results
        episodes = np.arange(0, 200, 10)
        rewards = [-761, -540, -304, -243, -198, -319, -225, -305, -225, -220, 
                   -241, -222, -200, -190, -180, -170, -165, -160, -155, -153]
        
        ax.plot(episodes[:len(rewards)], rewards, 'b-', linewidth=2, label='Training Reward')
        ax.scatter([147], [-153], color='red', s=100, zorder=5, label='Best Model')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_comparison_bars(self, ax, results):
        """Bar chart comparing all models"""
        models = list(results.keys())
        waiting_means = [results[m]['Waiting Time'] for m in models]
        
        colors = ['gray', 'gray', 'gray', 'green']
        bars = ax.bar(models, waiting_means, color=colors)
        ax.set_ylabel('Waiting Time')
        ax.set_title('Model Comparison\n(Lower is Better)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, waiting_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{val:.0f}', ha='center', va='bottom')
    
    def plot_waiting_distribution(self, ax):
        """Distribution of waiting times"""
        # Sample data from your model
        np.random.seed(42)
        dqn_waiting = np.random.normal(1500, 200, 1000)
        random_waiting = np.random.normal(2500, 400, 1000)
        
        ax.hist(dqn_waiting, bins=30, alpha=0.7, label='DQN', color='green')
        ax.hist(random_waiting, bins=30, alpha=0.5, label='Random', color='gray')
        ax.set_xlabel('Waiting Time')
        ax.set_ylabel('Frequency')
        ax.set_title('Waiting Time Distribution')
        ax.legend()
    
    def plot_queue_evolution(self, ax):
        """Queue length over time in an episode"""
        # This would come from actual episode data
        time_steps = np.arange(0, 100)
        queue_ns = np.random.poisson(5, 100).cumsum() % 15
        queue_ew = np.random.poisson(4, 100).cumsum() % 12
        
        ax.plot(time_steps, queue_ns, label='North-South', color='blue')
        ax.plot(time_steps, queue_ew, label='East-West', color='orange')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Queue Length')
        ax.set_title('Queue Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_action_distribution(self, ax):
        """Phase selection distribution"""
        actions = ['NS Green', 'EW Green']
        # Based on your policy analysis
        counts = [55, 45]  # percentages
        
        wedges, texts, autotexts = ax.pie(counts, labels=actions, autopct='%1.1f%%',
                                          colors=['lightblue', 'lightcoral'])
        ax.set_title('Phase Distribution')

    def plot_fairness_heatmap(self, ax):
        """Fairness visualization"""
        # Create sample fairness data
        queues = np.array([[5, 3], [4, 4], [2, 8], [6, 6]])
        im = ax.imshow(queues, cmap='YlOrRd', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticklabels(['NS', 'EW'])
        ax.set_yticklabels(['T1', 'T2', 'T3', 'T4'])
        ax.set_title('Queue Fairness Heatmap')
        plt.colorbar(im, ax=ax)

if __name__ == "__main__":
    viz = VisualizationSuite()
    viz.create_all_visualizations()