import numpy as np
from collections import deque

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
        self.throughput = 0
        self.fairness_scores = []
        
    def update(self, state, action, info):
        """Update metrics at each step"""
        queue_ns, queue_ew = state[0], state[1]
        wait_ns, wait_ew = state[2], state[3]
        
        # 1. Queue lengths
        self.queue_lengths.append(queue_ns + queue_ew)
        
        # 2. Current waiting time
        current_wait = wait_ns + wait_ew
        self.waiting_times.append(current_wait)
        
        # 3. Max waiting time (for any vehicle)
        self.max_waiting_times.append(max(wait_ns, wait_ew))
        
        # 4. Phase switches count
        if self.last_phase is not None and action != self.last_phase:
            self.phase_switches += 1
        self.last_phase = action
        
        # 5. Fairness (how balanced are queues)
        if queue_ns + queue_ew > 0:
            fairness = 1 - abs(queue_ns - queue_ew) / (queue_ns + queue_ew)
            self.fairness_scores.append(fairness)
        
        # 6. Throughput from info
        self.throughput = info.get('vehicles_passed', 0)
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'avg_queue_length': np.mean(self.queue_lengths),
            'max_queue_length': np.max(self.queue_lengths),
            'avg_waiting_time': np.mean(self.waiting_times),
            'max_waiting_time': np.max(self.max_waiting_times),
            'total_phase_switches': self.phase_switches,
            'avg_fairness': np.mean(self.fairness_scores) if self.fairness_scores else 0,
            'total_throughput': self.throughput
        }
    
    def print_metrics(self):
        """Print formatted metrics"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("🚦 TRAFFIC METRICS SUMMARY")
        print("="*50)
        print(f"📊 Average Queue Length:    {summary['avg_queue_length']:.2f} vehicles")
        print(f"📊 Maximum Queue Length:    {summary['max_queue_length']:.2f} vehicles")
        print(f"⏱️  Average Waiting Time:    {summary['avg_waiting_time']:.2f} sec")
        print(f"⏱️  Maximum Waiting Time:    {summary['max_waiting_time']:.2f} sec")
        print(f"🔄 Phase Switches:          {summary['total_phase_switches']}")
        print(f"⚖️  Fairness Score:          {summary['avg_fairness']:.3f}")
        print(f"🚗 Total Throughput:         {summary['total_throughput']} vehicles")
        print("="*50)