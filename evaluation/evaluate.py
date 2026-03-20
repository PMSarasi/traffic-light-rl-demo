"""
Traffic Light RL Evaluation Script
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ADD THESE MISSING IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import datetime
import argparse
from collections import defaultdict

# Now your existing imports
from environment.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQNAgent

# ============================================
# Advanced Metrics Class
# ============================================
class TrafficMetrics:
    """Real-world traffic metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.queue_lengths = []
        self.waiting_times = []
        self.max_waiting_times = []
        self.phase_switches = 0
        self.last_phase = None
        self.fairness_scores = []
        self.throughput = 0
    
    def update(self, state, action, info):
        """Update metrics at each step"""
        queue_ns, queue_ew = state[0], state[1]
        wait_ns, wait_ew = state[2], state[3]
        
        # Queue lengths
        self.queue_lengths.append(queue_ns + queue_ew)
        
        # Waiting times
        self.waiting_times.append(wait_ns + wait_ew)
        self.max_waiting_times.append(max(wait_ns, wait_ew))
        
        # Phase switches
        if self.last_phase is not None and action != self.last_phase:
            self.phase_switches += 1
        self.last_phase = action
        
        # Fairness (how balanced are queues)
        if queue_ns + queue_ew > 0:
            fairness = 1 - abs(queue_ns - queue_ew) / (queue_ns + queue_ew)
            self.fairness_scores.append(fairness)
        
        self.throughput = info.get('vehicles_passed', 0)
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'avg_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': np.max(self.queue_lengths) if self.queue_lengths else 0,
            'avg_waiting_time': np.mean(self.waiting_times) if self.waiting_times else 0,
            'max_waiting_time': np.max(self.max_waiting_times) if self.max_waiting_times else 0,
            'total_phase_switches': self.phase_switches,
            'avg_fairness': np.mean(self.fairness_scores) if self.fairness_scores else 0
        }


# ============================================
# Multi-Model Comparison (FIXED for Gymnasium API)
# ============================================
def random_policy(env, num_episodes=50):
    """Baseline 1: Random actions"""
    results = []
    metrics_collector = TrafficMetrics()
    
    for ep in range(num_episodes):
        # Handle reset return value (Gymnasium returns tuple)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        done = False
        metrics_collector.reset()
        
        while not done:
            action = np.random.randint(0, 2)
            # Handle step return values (Gymnasium returns 5 values)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            metrics_collector.update(state, action, info)
            state = next_state
        
        summary = metrics_collector.get_summary()
        summary['reward'] = reward
        summary['throughput'] = info['vehicles_passed']
        results.append(summary)
    
    return results

def fixed_time_policy(env, num_episodes=50, green_duration=30):
    """Baseline 2: Fixed-time traffic light"""
    results = []
    metrics_collector = TrafficMetrics()
    
    for ep in range(num_episodes):
        # Handle reset return value
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        done = False
        steps = 0
        metrics_collector.reset()
        
        while not done:
            action = 0 if (steps // green_duration) % 2 == 0 else 1
            # Handle step return values
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            metrics_collector.update(state, action, info)
            state = next_state
            steps += 1
        
        summary = metrics_collector.get_summary()
        summary['reward'] = reward
        summary['throughput'] = info['vehicles_passed']
        results.append(summary)
    
    return results

def rule_based_policy(env, num_episodes=50):
    """Baseline 3: Serve longer queue first"""
    results = []
    metrics_collector = TrafficMetrics()
    
    for ep in range(num_episodes):
        # Handle reset return value
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        done = False
        metrics_collector.reset()
        
        while not done:
            action = 0 if state[0] > state[1] else 1  # Compare queues
            # Handle step return values
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            metrics_collector.update(state, action, info)
            state = next_state
        
        summary = metrics_collector.get_summary()
        summary['reward'] = reward
        summary['throughput'] = info['vehicles_passed']
        results.append(summary)
    
    return results


# ============================================
# Enhanced Evaluation with Pipeline
# ============================================
class EvaluationPipeline:
    """Complete evaluation pipeline"""
    
    def __init__(self, output_dir='evaluation_results'):
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        self.data_dir = os.path.join(output_dir, 'data')
        self.reports_dir = os.path.join(output_dir, 'reports')
        
        for dir_path in [self.figures_dir, self.data_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_comparison(self, num_episodes=50):
        """Run all models and compare"""
        print("\n" + "="*60)
        print("🔄 Running Multi-Model Comparison...")
        print("="*60)
        
        env = TrafficLightEnv()
        
        # Run all policies
        print(f"\n📊 Random Policy (Baseline)...")
        random_results = random_policy(env, num_episodes)
        
        print(f"📊 Fixed-Time Policy (Traditional)...")
        fixed_results = fixed_time_policy(env, num_episodes)
        
        print(f"📊 Rule-Based Policy (Heuristic)...")
        rule_results = rule_based_policy(env, num_episodes)
        
        print(f"📊 DQN Model (Your Agent)...")
        dqn_results = self._evaluate_dqn(env, num_episodes)
        
        # Compile summary
        summary = {
            'Random': self._summarize_results(random_results),
            'Fixed-Time': self._summarize_results(fixed_results),
            'Rule-Based': self._summarize_results(rule_results),
            'DQN': self._summarize_results(dqn_results)
        }
        
        return summary
    
    def _evaluate_dqn(self, env, num_episodes):
        """Evaluate DQN agent"""
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        
        model_path = 'models/best_model.pt'
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"⚠️ Model not found at {model_path}, using random policy")
            return random_policy(env, num_episodes)
        
        results = []
        metrics_collector = TrafficMetrics()
        
        for ep in range(num_episodes):
            # Handle reset return value
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
            else:
                state = reset_result
                
            done = False
            metrics_collector.reset()
            
            while not done:
                action = agent.act(state, eval_mode=True)
                # Handle step return values
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result
                
                metrics_collector.update(state, action, info)
                state = next_state
            
            summary = metrics_collector.get_summary()
            summary['reward'] = reward
            summary['throughput'] = info['vehicles_passed']
            results.append(summary)
        
        return results
    
    def _summarize_results(self, results):
        """Calculate statistics from results"""
        waiting_times = [r['avg_waiting_time'] for r in results]
        throughputs = [r['throughput'] for r in results]
        rewards = [r['reward'] for r in results]
        
        return {
            'Waiting Time': np.mean(waiting_times),
            'Waiting Time Std': np.std(waiting_times),
            'Throughput': np.mean(throughputs),
            'Reward': np.mean(rewards),
            'Max Queue': np.mean([r['max_queue_length'] for r in results]),
            'Fairness': np.mean([r['avg_fairness'] for r in results]),
            'Phase Switches': np.mean([r['total_phase_switches'] for r in results])
        }
    
    def plot_comparison(self, summary, save_path=None):
        """Enhanced comparison plot"""
        if save_path is None:
            save_path = os.path.join(self.figures_dir, 'model_comparison.png')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Traffic Light RL - Model Comparison', fontsize=16, fontweight='bold')
        
        models = list(summary.keys())
        x_pos = np.arange(len(models))
        
        # Plot 1: Waiting Time
        waiting_means = [summary[m]['Waiting Time'] for m in models]
        waiting_stds = [summary[m]['Waiting Time Std'] for m in models]
        
        axes[0,0].bar(x_pos, waiting_means, yerr=waiting_stds, capsize=5, 
                      color=['gray', 'gray', 'gray', 'green'])
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(models, rotation=45)
        axes[0,0].set_ylabel('Waiting Time')
        axes[0,0].set_title('Average Waiting Time (Lower is Better)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput
        throughput_means = [summary[m]['Throughput'] for m in models]
        axes[0,1].bar(x_pos, throughput_means, color=['gray', 'gray', 'gray', 'green'])
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(models, rotation=45)
        axes[0,1].set_ylabel('Vehicles Passed')
        axes[0,1].set_title('Throughput (Higher is Better)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Max Queue
        max_queue_means = [summary[m]['Max Queue'] for m in models]
        axes[0,2].bar(x_pos, max_queue_means, color=['gray', 'gray', 'gray', 'green'])
        axes[0,2].set_xticks(x_pos)
        axes[0,2].set_xticklabels(models, rotation=45)
        axes[0,2].set_ylabel('Max Queue Length')
        axes[0,2].set_title('Peak Queue Length')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Fairness
        fairness_means = [summary[m]['Fairness'] for m in models]
        axes[1,0].bar(x_pos, fairness_means, color=['gray', 'gray', 'gray', 'green'])
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(models, rotation=45)
        axes[1,0].set_ylabel('Fairness Score')
        axes[1,0].set_title('Fairness (1=Perfect Balance)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Phase Switches
        switch_means = [summary[m]['Phase Switches'] for m in models]
        axes[1,1].bar(x_pos, switch_means, color=['gray', 'gray', 'gray', 'green'])
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(models, rotation=45)
        axes[1,1].set_ylabel('Phase Switches')
        axes[1,1].set_title('Switching Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Improvement Percentage
        baseline = waiting_means[0]  # Random as baseline
        improvements = [(baseline - w)/baseline * 100 for w in waiting_means]
        colors = ['gray', 'gray', 'gray', 'green' if improvements[3] > 0 else 'red']
        axes[1,2].bar(x_pos, improvements, color=colors)
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(models, rotation=45)
        axes[1,2].set_ylabel('Improvement %')
        axes[1,2].set_title('Improvement vs Random')
        axes[1,2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Comparison plot saved to: {save_path}")
        plt.show()
    
    def generate_report(self, summary, save_path=None):
        """Generate markdown report"""
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.reports_dir, f'report_{timestamp}.md')
        
        with open(save_path, 'w') as f:
            f.write("# Traffic Light RL - Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## 1. Model Comparison Summary\n\n")
            f.write("| Model | Waiting Time | Throughput | Fairness | Max Queue |\n")
            f.write("|-------|--------------|------------|----------|-----------|\n")
            
            for model, metrics in summary.items():
                f.write(f"| {model} | {metrics['Waiting Time']:.1f} ± {metrics['Waiting Time Std']:.1f} | "
                       f"{metrics['Throughput']:.1f} | {metrics['Fairness']:.3f} | "
                       f"{metrics['Max Queue']:.1f} |\n")
            
            # Calculate improvement
            baseline = summary['Random']['Waiting Time']
            dqn = summary['DQN']['Waiting Time']
            improvement = ((baseline - dqn) / baseline) * 100
            
            f.write(f"\n## 2. Key Findings\n\n")
            f.write(f"* **DQN Improvement:** {improvement:.1f}% better waiting time than random\n")
            f.write(f"* **Best Model Achieved:** -153.38 reward at episode 147 (from training)\n")
            f.write(f"* **Fairness Score:** {summary['DQN']['Fairness']:.3f} (closer to 1 = better)\n")
            f.write(f"* **Phase Switches:** {summary['DQN']['Phase Switches']:.1f} per episode\n\n")
            
            f.write("## 3. Visualizations\n\n")
            f.write("The following figures were generated:\n")
            f.write("* `model_comparison.png` - Comprehensive model comparison\n")
            f.write("* `evaluation_results.png` - Basic evaluation plots\n")
        
        print(f"📄 Report saved to: {save_path}")
        return save_path
    
    def run_pipeline(self, num_episodes=50):
        """Run complete evaluation pipeline"""
        print("\n" + "="*60)
        print("🚦 TRAFFIC LIGHT RL - FULL EVALUATION PIPELINE")
        print("="*60)
        print(f"Evaluating {num_episodes} episodes per model\n")
        
        # Step 1: Multi-model comparison
        print("📊 STEP 1/4: Running Multi-Model Comparison...")
        summary = self.run_comparison(num_episodes)
        
        # Step 2: Save results
        print("\n💾 STEP 2/4: Saving results...")
        results_path = os.path.join(self.data_dir, 'comparison_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy values to Python types
            serializable = {}
            for model, metrics in summary.items():
                serializable[model] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()
                }
            json.dump(serializable, f, indent=2)
        print(f"   Results saved to {results_path}")
        
        # Step 3: Generate visualizations
        print("\n📈 STEP 3/4: Generating visualizations...")
        self.plot_comparison(summary)
        
        # Step 4: Create report
        print("\n📝 STEP 4/4: Creating report...")
        report_path = self.generate_report(summary)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("="*60)
        
        return summary


# ============================================
# Original evaluate_model function (FIXED for Gymnasium API)
# ============================================
def evaluate_model(model_path='models/best_model.pt', num_episodes=50):
    """Comprehensive evaluation of trained model"""
    
    # Load environment and agent
    env = TrafficLightEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Evaluation metrics
    metrics = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'queue_lengths': [],
        'throughput': []
    }
    
    # Compare with fixed-time controller
    fixed_time_results = evaluate_fixed_time(env, num_episodes)
    
    # Evaluate RL agent
    for episode in range(num_episodes):
        # Handle reset return value
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        episode_reward = 0
        queue_lengths = []
        done = False
        
        while not done:
            action = agent.act(state, eval_mode=True)
            # Handle step return values
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
                
            episode_reward += reward
            queue_lengths.append((info['queue_ns'], info['queue_ew']))
            state = next_state
        
        metrics['episode_rewards'].append(episode_reward)
        metrics['avg_waiting_times'].append(info['waiting_time'])
        metrics['queue_lengths'].append(np.mean(queue_lengths))
        metrics['throughput'].append(info['vehicles_passed'])
    
    # Print results
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"RL Agent - Avg Reward: {np.mean(metrics['episode_rewards']):.2f}")
    print(f"RL Agent - Avg Waiting Time: {np.mean(metrics['avg_waiting_times']):.2f}")
    print(f"RL Agent - Avg Queue Length: {np.mean(metrics['queue_lengths']):.2f}")
    print(f"RL Agent - Throughput: {np.mean(metrics['throughput']):.2f}")
    print("-"*50)
    print(f"Fixed-Time - Avg Waiting Time: {np.mean(fixed_time_results):.2f}")
    print(f"Improvement: {(np.mean(fixed_time_results) - np.mean(metrics['avg_waiting_times'])) / np.mean(fixed_time_results) * 100:.1f}%")
    print("="*50)
    
    # Plot comparison
    plot_comparison(metrics, fixed_time_results)
    
    return metrics

def evaluate_fixed_time(env, num_episodes=50):
    """Evaluate fixed-time traffic light controller"""
    waiting_times = []
    
    for _ in range(num_episodes):
        # Handle reset return value
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
            
        done = False
        phase_time = 0
        
        while not done:
            # Fixed cycle: 30 seconds each phase
            action = 0 if phase_time < 30 else 1
            # Handle step return values
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
                
            state = next_state
            phase_time = (phase_time + 5) % 60
        
        waiting_times.append(info['waiting_time'])
    
    return waiting_times

def plot_comparison(metrics, fixed_time_results):
    """Plot comparison between RL and fixed-time"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Waiting time comparison
    axes[0,0].bar(['RL Agent', 'Fixed-Time'], 
                   [np.mean(metrics['avg_waiting_times']), np.mean(fixed_time_results)],
                   color=['green', 'red'])
    axes[0,0].set_ylabel('Average Waiting Time')
    axes[0,0].set_title('Waiting Time Comparison')
    axes[0,0].grid(True, alpha=0.3)
    
    # Queue lengths over time
    axes[0,1].plot(metrics['queue_lengths'], label='RL Agent', color='green')
    axes[0,1].axhline(y=np.mean(metrics['queue_lengths']), color='green', 
                      linestyle='--', label='RL Avg')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Average Queue Length')
    axes[0,1].set_title('Queue Lengths')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Throughput
    axes[1,0].hist(metrics['throughput'], bins=20, alpha=0.7, color='blue')
    axes[1,0].axvline(x=np.mean(metrics['throughput']), color='red', 
                      linestyle='--', label=f"Mean: {np.mean(metrics['throughput']):.1f}")
    axes[1,0].set_xlabel('Vehicles Passed')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Throughput Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Rewards
    axes[1,1].plot(metrics['episode_rewards'], color='purple')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Reward')
    axes[1,1].set_title('Evaluation Rewards')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()


# ============================================
# Main execution with argument parsing
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic Light RL Evaluation')
    parser.add_argument('--mode', type=str, choices=['basic', 'pipeline', 'comparison'], 
                       default='basic', help='Evaluation mode')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'pipeline':
        # Run full pipeline
        pipeline = EvaluationPipeline()
        pipeline.run_pipeline(num_episodes=args.episodes)
    
    elif args.mode == 'comparison':
        # Just run comparison without full pipeline
        pipeline = EvaluationPipeline()
        summary = pipeline.run_comparison(num_episodes=args.episodes)
        pipeline.plot_comparison(summary)
        
        # Print summary table
        df = pd.DataFrame(summary).T
        print("\n📊 Comparison Summary:")
        print(df.round(2))
    
    else:
        # Basic evaluation (your original)
        evaluate_model(model_path=args.model, num_episodes=args.episodes)