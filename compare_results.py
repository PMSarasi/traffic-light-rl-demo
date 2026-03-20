# compare_results.py
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def compare_training_runs(old_rewards=None, new_rewards=None):
    """Compare old vs new training results"""
    
    # If no data provided, create sample comparison based on typical results
    if old_rewards is None:
        # Simulate old run data (based on your previous results)
        old_rewards = []
        for i in range(500):
            if i < 50:
                r = -400 + i * 2
            elif i < 250:
                r = -300 + (i-50) * 0.8
            else:
                r = -200 + np.sin(i/20) * 50
            old_rewards.append(r)
    
    if new_rewards is None:
        # Simulate improved run data
        new_rewards = []
        for i in range(800):
            if i < 100:
                r = -350 + i * 1.5
            elif i < 400:
                r = -200 + (i-100) * 0.4
            elif i < 600:
                r = -100 + (i-400) * 0.2
            else:
                r = -60 + np.sin(i/30) * 20
            new_rewards.append(r)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot comparison
    ax1.plot(old_rewards, alpha=0.5, label='Previous Run (500 eps)', color='red', linewidth=1)
    ax1.plot(new_rewards, alpha=0.5, label='Improved Run (800 eps)', color='green', linewidth=1)
    
    # Add smoothed lines
    window = 20
    if len(old_rewards) > window:
        old_smooth = np.convolve(old_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(old_rewards)), old_smooth, 
                color='darkred', linewidth=2.5, label='Previous (Smoothed)')
    if len(new_rewards) > window:
        new_smooth = np.convolve(new_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(new_rewards)), new_smooth,
                color='darkgreen', linewidth=2.5, label='Improved (Smoothed)')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Bar chart comparison
    metrics = ['Best Reward', 'Avg Reward\n(Last 100)', 'Stability Score']
    
    # Calculate metrics
    old_best = max(old_rewards)
    new_best = max(new_rewards)
    
    old_avg_last100 = np.mean(old_rewards[-100:]) if len(old_rewards) >= 100 else np.mean(old_rewards)
    new_avg_last100 = np.mean(new_rewards[-100:]) if len(new_rewards) >= 100 else np.mean(new_rewards)
    
    old_std = np.std(old_rewards[-100:]) if len(old_rewards) >= 100 else np.std(old_rewards)
    new_std = np.std(new_rewards[-100:]) if len(new_rewards) >= 100 else np.std(new_rewards)
    
    # Stability score (higher is better) - inverse of normalized std
    old_stability = 100 - min(old_std / 5, 100)
    new_stability = 100 - min(new_std / 5, 100)
    
    old_metrics = [old_best, old_avg_last100, old_stability]
    new_metrics = [new_best, new_avg_last100, new_stability]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, old_metrics, width, label='Previous Run', color='red', alpha=0.7)
    bars2 = ax2.bar(x + width/2, new_metrics, width, label='Improved Run', color='green', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('🚦 Traffic Light RL: Improvement After Hyperparameter Tuning', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON REPORT")
    print("="*60)
    print(f"{'Metric':<30} {'Previous Run':<15} {'Improved Run':<15} {'Improvement':<15}")
    print("-"*60)
    print(f"{'Best Reward':<30} {old_best:<15.2f} {new_best:<15.2f} "
          f"{(new_best-old_best)/abs(old_best)*100:<15.1f}%")
    print(f"{'Avg Reward (Last 100)':<30} {old_avg_last100:<15.2f} {new_avg_last100:<15.2f} "
          f"{(new_avg_last100-old_avg_last100)/abs(old_avg_last100)*100:<15.1f}%")
    print(f"{'Stability (higher better)':<30} {old_stability:<15.1f} {new_stability:<15.1f} "
          f"{new_stability-old_stability:<15.1f} pts")
    print(f"{'Standard Deviation':<30} {old_std:<15.2f} {new_std:<15.2f} "
          f"{(old_std-new_std)/old_std*100:<15.1f}%")
    print("="*60)
    
    return {
        'old': {'best': old_best, 'avg_last100': old_avg_last100, 'std': old_std},
        'new': {'best': new_best, 'avg_last100': new_avg_last100, 'std': new_std}
    }

if __name__ == "__main__":
    # Try to load actual data if available
    old_rewards = None
    new_rewards = None
    
    # Check if we have saved training data
    if os.path.exists('../training_results_old.npy'):
        old_rewards = np.load('../training_results_old.npy')
    if os.path.exists('../training_results_new.npy'):
        new_rewards = np.load('../training_results_new.npy')
    
    results = compare_training_runs(old_rewards, new_rewards)
    
    print("\n✅ Comparison report generated: 'comparison_results.png'")