"""
Traffic Light RL - Simple Demo UI
Run this to see your model in action!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

from environment.traffic_env import TrafficLightEnv
from agents.dqn_agent import DQNAgent

class TrafficLightDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Traffic Light Control - DQN Demo")
        self.root.geometry("1000x700")
        
        self.env = None
        self.agent = None
        self.running = False
        
        # Load model
        self.load_model()
        
        self.setup_ui()
        
    def load_model(self):
        """Load trained DQN model"""
        try:
            self.agent = DQNAgent(7, 2)
            model_path = 'models/best_model.pt'
            if os.path.exists(model_path):
                self.agent.load(model_path)
                print("✅ Model loaded successfully!")
            else:
                print("⚠️ Model not found, using random policy")
        except:
            print("❌ Error loading model")
            self.agent = None
    
    def setup_ui(self):
        """Create UI elements"""
        
        # Title
        title = tk.Label(self.root, text="🚦 Intelligent Traffic Light Control", 
                        font=("Arial", 20, "bold"), fg="darkgreen")
        title.pack(pady=10)
        
        # Control Frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Model Selection
        tk.Label(control_frame, text="Model:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar(value="DQN")
        model_menu = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                  values=["DQN", "Fixed-Time", "Rule-Based", "Random"], 
                                  state="readonly", width=15)
        model_menu.grid(row=0, column=1, padx=5)
        
        # Traffic Level
        tk.Label(control_frame, text="Traffic:", font=("Arial", 12)).grid(row=0, column=2, padx=5)
        self.traffic_var = tk.StringVar(value="Medium")
        traffic_menu = ttk.Combobox(control_frame, textvariable=self.traffic_var,
                                    values=["Low", "Medium", "High"],
                                    state="readonly", width=10)
        traffic_menu.grid(row=0, column=3, padx=5)
        
        # Run Button
        self.run_btn = tk.Button(control_frame, text="▶ RUN", command=self.run_simulation,
                                 bg="green", fg="white", font=("Arial", 12, "bold"), width=10)
        self.run_btn.grid(row=0, column=4, padx=20)
        
        # Stop Button
        self.stop_btn = tk.Button(control_frame, text="⏹ STOP", command=self.stop_simulation,
                                  bg="red", fg="white", font=("Arial", 12, "bold"), width=10, state="disabled")
        self.stop_btn.grid(row=0, column=5, padx=5)
        
        # Metrics Frame
        metrics_frame = tk.Frame(self.root, relief=tk.RIDGE, bd=2)
        metrics_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(metrics_frame, text="📊 LIVE METRICS", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=4, pady=5)
        
        self.waiting_label = tk.Label(metrics_frame, text="Waiting Time: --", font=("Arial", 12))
        self.waiting_label.grid(row=1, column=0, padx=20, pady=5)
        
        self.queue_label = tk.Label(metrics_frame, text="Queue: NS:-- EW:--", font=("Arial", 12))
        self.queue_label.grid(row=1, column=1, padx=20, pady=5)
        
        self.throughput_label = tk.Label(metrics_frame, text="Throughput: --", font=("Arial", 12))
        self.throughput_label.grid(row=1, column=2, padx=20, pady=5)
        
        self.phase_label = tk.Label(metrics_frame, text="Phase: --", font=("Arial", 12))
        self.phase_label.grid(row=1, column=3, padx=20, pady=5)
        
        # Graph Frame
        graph_frame = tk.Frame(self.root)
        graph_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.suptitle("Traffic Status")
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax1.set_title("Queue Lengths")
        self.ax1.set_ylabel("Vehicles")
        self.ax2.set_title("Waiting Time")
        self.ax2.set_ylabel("Time")
        
        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def get_action(self, state, model):
        """Get action based on selected model"""
        if model == "Random":
            return np.random.randint(0, 2)
        elif model == "Fixed-Time":
            # Simple fixed timing
            return 0 if (time.time() % 60) < 30 else 1
        elif model == "Rule-Based":
            return 0 if state[0] > state[1] else 1
        else:  # DQN
            if self.agent:
                return self.agent.act(state, eval_mode=True)
            return np.random.randint(0, 2)
    
    def run_simulation(self):
        """Run simulation in separate thread"""
        if self.running:
            return
        
        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_bar.config(text="Running simulation...")
        
        thread = threading.Thread(target=self._simulate)
        thread.daemon = True
        thread.start()
    
    def _simulate(self):
        """Simulation logic"""
        env = TrafficLightEnv()
        model = self.model_var.get()
        
        # Adjust traffic based on selection
        traffic_level = self.traffic_var.get()
        
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        queue_history = []
        waiting_history = []
        
        for step in range(100):  # Run 100 steps
            if not self.running:
                break
            
            action = self.get_action(state, model)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            total_reward += reward
            steps += 1
            queue_history.append([info['queue_ns'], info['queue_ew']])
            waiting_history.append(info['waiting_time'])
            
            # Update UI
            self.root.after(0, self._update_ui, info, state, model, action)
            
            # Update graphs every 10 steps
            if step % 10 == 0:
                self.root.after(0, self._update_graphs, queue_history, waiting_history)
            
            state = next_state
            
            if terminated or truncated:
                break
            
            time.sleep(0.1)  # Slow down for visibility
        
        self.running = False
        self.root.after(0, self._simulation_done, total_reward, steps, info)
    
    def _update_ui(self, info, state, model, action):
        """Update UI with live metrics"""
        phase = "🟢 NS Green" if action == 0 else "🟢 EW Green"
        self.phase_label.config(text=f"Phase: {phase}")
        self.waiting_label.config(text=f"Waiting Time: {info['waiting_time']:.0f}")
        self.queue_label.config(text=f"Queue: NS:{info['queue_ns']} EW:{info['queue_ew']}")
        self.throughput_label.config(text=f"Throughput: {info['vehicles_passed']}")
    
    def _update_graphs(self, queue_history, waiting_history):
        """Update matplotlib graphs"""
        self.ax1.clear()
        self.ax2.clear()
        
        if queue_history:
            queues = np.array(queue_history)
            self.ax1.plot(queues[:, 0], label='NS', color='blue')
            self.ax1.plot(queues[:, 1], label='EW', color='orange')
            self.ax1.set_title("Queue Lengths")
            self.ax1.set_ylabel("Vehicles")
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
        
        if waiting_history:
            self.ax2.plot(waiting_history, color='red')
            self.ax2.set_title("Waiting Time")
            self.ax2.set_ylabel("Time")
            self.ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _simulation_done(self, total_reward, steps, info):
        """Handle simulation completion"""
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_bar.config(text=f"✅ Complete! Reward: {total_reward:.2f}, Waiting: {info['waiting_time']}, Throughput: {info['vehicles_passed']}")
    
    def stop_simulation(self):
        """Stop running simulation"""
        self.running = False
        self.status_bar.config(text="Stopped")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficLightDemo(root)
    root.mainloop()