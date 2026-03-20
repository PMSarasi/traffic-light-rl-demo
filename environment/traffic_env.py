# environment/traffic_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class TrafficLightEnv(gym.Env):
    """Custom Traffic Light Control Environment"""
    
    def __init__(self, config=None, render_mode=None):
        super(TrafficLightEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # Default configuration
        self.config = config or {
            'num_lanes': 4,
            'max_queue': 20,
            'time_step': 5,
            'yellow_time': 3,
            'min_green': 10,
            'max_green': 60
        }
        
        # Action space: 0=NS green, 1=EW green
        self.action_space = spaces.Discrete(2)
        
        # OBSERVATION SPACE - IMPROVED with more features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, -20]),
            high=np.array([self.config['max_queue'], self.config['max_queue'], 
                          300, 300, 1, self.config['max_green'], 20]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.queue_ns = np.random.randint(0, 5)
        self.queue_ew = np.random.randint(0, 5)
        self.waiting_time_ns = 0
        self.waiting_time_ew = 0
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_waiting_time = 0
        self.vehicles_passed = 0
        self.time_since_last_switch = 0
        self.action_changed = False  # ADD THIS LINE
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Return current state"""
        queue_imbalance = self.queue_ns - self.queue_ew
        
        return np.array([
            self.queue_ns,
            self.queue_ew,
            self.waiting_time_ns,
            self.waiting_time_ew,
            self.current_phase,
            self.time_in_phase,
            queue_imbalance
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute action"""
        # ADD THIS LINE - Track if action changed
        self.action_changed = (action != self.current_phase)
        
        # ACTION CONSTRAINT - Prevent rapid switching
        if action != self.current_phase:
            if self.time_in_phase < self.config['min_green']:
                action = self.current_phase
            else:
                self.time_since_last_switch = 0
        else:
            self.time_since_last_switch += self.config['time_step']
        
        # Update time in phase
        if action == self.current_phase:
            self.time_in_phase += self.config['time_step']
        else:
            self.time_in_phase = 0
            self.current_phase = action
        
        # Simulate traffic flow
        self._update_traffic(action)
        
        # Calculate reward - NEW IMPROVED VERSION
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._check_done()
        truncated = False
        
        # Optional rendering
        if self.render_mode == "human":
            self.render()
        
        return self._get_state(), reward, terminated, truncated, {
            'queue_ns': self.queue_ns,
            'queue_ew': self.queue_ew,
            'waiting_time': self.total_waiting_time,
            'vehicles_passed': self.vehicles_passed
        }
    
    def _update_traffic(self, action):
        """Update traffic based on current green phase"""
        # New vehicles arrive (random)
        self.queue_ns += np.random.poisson(2) if np.random.random() > 0.3 else 0
        self.queue_ew += np.random.poisson(2) if np.random.random() > 0.3 else 0
        
        # Vehicles pass based on green phase
        if action == 0:  # NS green
            passed = min(self.queue_ns, np.random.randint(3, 6))
            self.queue_ns -= passed
            self.vehicles_passed += passed
            self.waiting_time_ew += self.queue_ew
        else:  # EW green
            passed = min(self.queue_ew, np.random.randint(3, 6))
            self.queue_ew -= passed
            self.vehicles_passed += passed
            self.waiting_time_ns += self.queue_ns
        
        # Update waiting times
        self.waiting_time_ns += self.queue_ns
        self.waiting_time_ew += self.queue_ew
        self.total_waiting_time = self.waiting_time_ns + self.waiting_time_ew
    
    def _calculate_reward(self):
        """Calculate reward - IMPROVED VERSION with throughput encouragement"""
        
        # Current metrics
        total_waiting = self.waiting_time_ns + self.waiting_time_ew
        max_queue = max(self.queue_ns, self.queue_ew)
        queue_imbalance = abs(self.queue_ns - self.queue_ew)
        
        # ========== PENALTIES (Negative) ==========
        
        # Base waiting penalty
        waiting_penalty = total_waiting * 1.0
        
        # Strong queue penalty (FIX 4)
        queue_penalty = 2.0 * max_queue
        
        # Fairness penalty (FIX 5)
        fairness_penalty = 0.8 * queue_imbalance
        
        # Green duration penalty - prevents stuck signals (FIX 2)
        green_penalty = 0.1 * self.time_in_phase
        
        # ========== BONUSES (Positive) ==========
        
        # Throughput bonus - CRITICAL for traffic flow! (FIX 1)
        throughput_bonus = 2.0 * self.vehicles_passed
        
        # Switching bonus - encourages alternating lights (FIX 3)
        switch_bonus = 1.0 if self.action_changed else 0
        
        # Bonus for clearing queues completely
        if self.queue_ns == 0 and self.queue_ew == 0:
            throughput_bonus += 3.0
        
        # ========== COMBINE ==========
        
        # All penalties are negative, all bonuses are positive
        reward = -(
            waiting_penalty +
            queue_penalty +
            fairness_penalty +
            green_penalty
        ) + throughput_bonus + switch_bonus
        
        # Normalize to reasonable range
        reward = reward / 200.0
        
        # Optional: clip extreme values
        reward = max(min(reward, 10), -10)
        
        return reward
    
    def _check_done(self):
        """Check if episode should end"""
        if self.total_waiting_time > 2000 or self.vehicles_passed > 800:
            return True
        return False
    
    def render(self):
        """Visualize the traffic state"""
        if self.render_mode == "human":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Traffic light status
            colors = ['green' if self.current_phase == 0 else 'red',
                      'green' if self.current_phase == 1 else 'red']
            
            # Draw intersection
            ax1.set_xlim(0, 10)
            ax1.set_ylim(0, 10)
            ax1.set_aspect('equal')
            
            # Roads
            ax1.add_patch(Rectangle((0, 4), 10, 2, fill=True, alpha=0.3, color='gray'))
            ax1.add_patch(Rectangle((4, 0), 2, 10, fill=True, alpha=0.3, color='gray'))
            
            # Traffic lights
            ax1.text(4.5, 4.5, 'NS', ha='center', va='center', fontsize=12)
            ax1.text(5.5, 5.5, f'●', color=colors[0], fontsize=20)
            ax1.text(4.5, 5.5, 'EW', ha='center', va='center', fontsize=12)
            ax1.text(5.5, 4.5, f'●', color=colors[1], fontsize=20)
            
            # Queue visualization
            ax2.bar(['NS Queue', 'EW Queue'], [self.queue_ns, self.queue_ew])
            ax2.set_ylim(0, 20)
            ax2.set_ylabel('Number of Vehicles')
            ax2.set_title('Current Queue Lengths')
            
            plt.tight_layout()
            plt.show()