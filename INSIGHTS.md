# 🧠 Key Insights - Why DQN Behaves This Way

## The Discovery

After comparing 4 models, here's what we learned:

### 📊 Model Comparison Summary

| Model | Strategy | Waiting Time | Throughput | Fairness |
|-------|----------|--------------|------------|----------|
| **Random** | Random switches | 951 | 466 | 0.23 |
| **Fixed-Time** | 30s each | 801 | 86 | 0.13 |
| **Rule-Based** | Longer queue wins | 1002 | 693 | 0.25 |
| **DQN** | Learned policy | **732** ✅ | 92 ❌ | 0.08 ❌ |

---

## 🎯 Why DQN Achieves LOW Waiting Time but POOR Throughput

### The "Cheating Strategy"

DQN discovered a mathematical loophole:
