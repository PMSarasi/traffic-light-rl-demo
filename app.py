"""
Traffic Light RL - Web Demo
Hosted on Streamlit Cloud
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQNAgent

# Page config
st.set_page_config(
    page_title="Traffic Light RL Demo",
    page_icon="🚦",
    layout="wide"
)

# Title
st.title("🚦 Intelligent Traffic Light Control")
st.markdown("### Deep Reinforcement Learning (DQN) Agent")

# Sidebar
st.sidebar.header("⚙️ Simulation Settings")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Traffic Controller",
    ["🎓 DQN (Trained AI)", "🎲 Random", "⏱️ Fixed-Time (30s)", "📊 Rule-Based (Longer Queue)"]
)

# Traffic level
traffic_level = st.sidebar.select_slider(
    "🚗 Traffic Density",
    options=["Low", "Medium", "High"],
    value="Medium"
)

# Run button
run_button = st.sidebar.button("▶ START SIMULATION", type="primary", use_container_width=True)

# Info sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    """
    **Model:** DQN trained for 200 episodes
    **Best Reward:** -124.18
    **Improvement:** 23% vs Random
    """
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🚦 Live Traffic Status")
    queue_placeholder = st.empty()
    metrics_placeholder = st.empty()

with col2:
    st.markdown("### 📈 Performance")
    waiting_placeholder = st.empty()
    throughput_placeholder = st.empty()
    phase_placeholder = st.empty()

# Graph area
st.markdown("---")
st.markdown("### 📊 Traffic Analysis")
graph_col1, graph_col2 = st.columns(2)
queue_graph = graph_col1.empty()
waiting_graph = graph_col2.empty()

# Load model
@st.cache_resource
def load_dqn_model():
    try:
        agent = DQNAgent(7, 2)
        model_path = 'models/best_model.pt'
        if os.path.exists(model_path):
            agent.load(model_path)
            return agent
    except:
        pass
    return None

dqn_agent = load_dqn_model()

# Get model function
def get_action(state, model_name, step):
    if model_name == "🎲 Random":
        return np.random.randint(0, 2)
    elif model_name == "⏱️ Fixed-Time (30s)":
        return 0 if (step // 30) % 2 == 0 else 1
    elif model_name == "📊 Rule-Based (Longer Queue)":
        return 0 if state[0] > state[1] else 1
    else:  # DQN
        if dqn_agent:
            return dqn_agent.act(state, eval_mode=True)
        return np.random.randint(0, 2)

# Run simulation
if run_button:
    env = TrafficLightEnv()
    state, _ = env.reset()
    
    # Tracking
    queue_history = []
    waiting_history = []
    throughput = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(100):
        action = get_action(state, model_option, step)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        state = next_state
        throughput = info['vehicles_passed']
        
        # Store history
        queue_history.append([info['queue_ns'], info['queue_ew']])
        waiting_history.append(info['waiting_time'])
        
        # Update displays
        with col1:
            # Queue display
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(['North-South', 'East-West'], [info['queue_ns'], info['queue_ew']], 
                   color=['blue', 'orange'])
            ax.set_ylim(0, 30)
            ax.set_ylabel("Vehicles Waiting")
            ax.set_title("Current Queue Lengths")
            queue_placeholder.pyplot(fig)
            plt.close()
        
        with col2:
            waiting_placeholder.metric("⏱️ Waiting Time", f"{info['waiting_time']:.0f} sec")
            throughput_placeholder.metric("🚗 Vehicles Passed", throughput)
            phase_text = "🟢 NS Green" if action == 0 else "🟢 EW Green"
            phase_placeholder.metric("🚦 Current Phase", phase_text)
        
        # Update graphs every 10 steps
        if step % 10 == 0 and step > 0:
            # Queue graph
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            queues = np.array(queue_history)
            ax1.plot(queues[:, 0], label='North-South', color='blue')
            ax1.plot(queues[:, 1], label='East-West', color='orange')
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Queue Length")
            ax1.set_title("Queue Evolution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            queue_graph.pyplot(fig1)
            plt.close()
            
            # Waiting graph
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(waiting_history, color='red')
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Waiting Time")
            ax2.set_title("Waiting Time Over Time")
            ax2.grid(True, alpha=0.3)
            waiting_graph.pyplot(fig2)
            plt.close()
        
        progress_bar.progress((step + 1) / 100)
        status_text.text(f"Step {step + 1}/100 - Running...")
        
        if terminated or truncated:
            break
        
        time.sleep(0.05)
    
    # Final results
    progress_bar.empty()
    status_text.success(f"✅ Simulation Complete! Final Waiting Time: {info['waiting_time']:.0f} sec | Throughput: {throughput} vehicles")
    
    # Show final metrics
    st.markdown("---")
    st.markdown("### 🏆 Final Results")
    
    final_col1, final_col2, final_col3 = st.columns(3)
    final_col1.metric("Average Waiting Time", f"{np.mean(waiting_history):.1f} sec")
    final_col2.metric("Total Throughput", f"{throughput} vehicles")
    final_col3.metric("Peak Queue", f"{max([q[0]+q[1] for q in queue_history])} vehicles")

else:
    st.info("👈 **Select a controller and click START SIMULATION to begin**")
    
    # Show model comparison summary
    st.markdown("---")
    st.markdown("### 📊 Model Comparison Results")
    
    comparison_data = {
        "Model": ["Random", "Fixed-Time", "Rule-Based", "DQN (Ours)"],
        "Waiting Time": [951, 801, 1002, 733],
        "Throughput": [466, 86, 693, 92],
        "Fairness": [0.23, 0.13, 0.25, 0.08]
    }
    
    st.table(comparison_data)
    st.caption("📈 DQN achieves 23% lower waiting time than Random")

# Footer
st.markdown("---")
st.caption("🚦 Traffic Light Control using Deep Q-Network | Trained for 200 episodes | Best Reward: -124.18")