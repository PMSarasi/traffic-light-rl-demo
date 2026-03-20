# training/train.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from environment.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQNAgent
from tqdm import tqdm
import yaml

def train_traffic_light(config_path=None):
    """Main training function"""
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'environment': {
                'num_lanes': 4,
                'max_queue': 20,
                'time_step': 5,
                'yellow_time': 3,
                'min_green': 10,
                'max_green': 60
            },
            'agent': {
                'learning_rate': 0.0005,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.05,
                'epsilon_decay': 0.999,        # CHANGED: slower decay
                'batch_size': 64,
                'memory_size': 20000,           # CHANGED: larger memory
                'target_update': 50              # CHANGED: less frequent
            },
            'training': {
                'episodes': 200,                 # CHANGED: from 1000 to 600 (optimal range)
                'max_steps': 150,
                'eval_interval': 25,              # CHANGED: more frequent evaluation
                'save_interval': 50
            }
        }
    
    # Initialize environment and agent
    env = TrafficLightEnv(config['environment'])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, config['agent'])
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    avg_waiting_times = []
    losses = []
    best_eval_score = -float('inf')  # NEW: Track best evaluation
    
    print("🚦 Starting Traffic Light RL Training...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Training for {config['training']['episodes']} episodes")
    print(f"Learning rate: {config['agent']['learning_rate']}")
    print(f"Target update: every {config['agent']['target_update']} episodes")
    print(f"Memory size: {config['agent']['memory_size']}")
    print(f"Epsilon decay: {config['agent']['epsilon_decay']}")
    
    # Training loop
    for episode in tqdm(range(config['training']['episodes'])):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(config['training']['max_steps']):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            
            loss = agent.learn()
            if loss:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update target network
        if episode % config['agent']['target_update'] == 0:
            agent.update_target_network()
            if episode > 0:
                print(f"\n🔄 Target network updated at episode {episode}")
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        avg_waiting_times.append(info['waiting_time'])
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # NEW: Save best model based on training reward
        agent.save_best_if_needed(episode_reward, episode)
        
        # Evaluation
        if episode % config['training']['eval_interval'] == 0:
            eval_score = evaluate(agent, env, num_episodes=5)
            
            # NEW: Track best evaluation score
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                agent.save('../models/best_eval_model.pt')
                print(f"\n⭐ NEW BEST EVAL! Episode {episode}, Eval Score: {eval_score:.2f}")
            
            print(f"\n📊 Episode {episode}: Train Reward: {episode_reward:.2f}, "
                  f"Eval Score: {eval_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if episode % config['training']['save_interval'] == 0 and episode > 0:
            os.makedirs('../models', exist_ok=True)
            agent.save(f'../models/checkpoint_ep{episode}.pt')
    
    # Save final model
    os.makedirs('../models', exist_ok=True)
    agent.save('../models/final_model.pt')
    
    # Print final summary
    print("\n" + "="*60)
    print("🏆 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best training reward: {max(episode_rewards):.2f} at episode {np.argmax(episode_rewards)}")
    print(f"Best evaluation score: {best_eval_score:.2f}")
    print(f"Best model saved as: ../models/best_model.pt")
    print(f"Final model saved as: ../models/final_model.pt")
    print("="*60)
    
    # Plot results with improved visualization
    plot_training_results(episode_rewards, avg_waiting_times, losses)
    
    return agent, episode_rewards

def evaluate(agent, env, num_episodes=5):
    """Evaluate agent performance"""
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

def plot_training_results(rewards, waiting_times, losses):
    """Plot training metrics with best model markers"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    window = 20
    if len(rewards) > window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_x = range(window-1, len(rewards))
    
    # 1. Rewards
    axes[0].plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    if len(rewards) > window:
        axes[0].plot(smoothed_x, smoothed_rewards, color='blue', linewidth=2, label='Smoothed (MA-20)')
    
    # NEW: Mark best episode
    best_idx = np.argmax(rewards)
    axes[0].plot(best_idx, rewards[best_idx], 'g*', markersize=15, 
                label=f'BEST Model: {rewards[best_idx]:.2f}')
    axes[0].axvline(x=best_idx, color='green', linestyle='--', alpha=0.3)
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards (Higher is Better)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Waiting times
    axes[1].plot(waiting_times, alpha=0.3, color='orange', label='Raw Waiting Time')
    if len(waiting_times) > window:
        smoothed_waiting = np.convolve(waiting_times, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed_x, smoothed_waiting, color='orange', linewidth=2, label='Smoothed (MA-20)')
    
    # NEW: Mark best waiting time
    best_wait_idx = np.argmin(waiting_times)
    axes[1].plot(best_wait_idx, waiting_times[best_wait_idx], 'g*', markersize=15,
                label=f'BEST: {waiting_times[best_wait_idx]:.2f}')
    axes[1].axvline(x=best_wait_idx, color='green', linestyle='--', alpha=0.3)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Waiting Time')
    axes[1].set_title('Average Waiting Time (Lower is Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Losses
    if losses:
        axes[2].plot(losses, color='red', label='Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss (Lower is Better)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Traffic Light RL Training Progress (Final Version)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../training_results_final.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    agent, rewards = train_traffic_light()