# test_imports.py
print("Testing imports...")

try:
    import numpy as np
    print("✅ numpy - OK")
except:
    print("❌ numpy - Failed")

try:
    import torch
    print("✅ torch - OK")
except:
    print("❌ torch - Failed")

try:
    import gymnasium as gym  
    print("✅ gym - OK")
except:
    print("❌ gym - Failed")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib - OK")
except:
    print("❌ matplotlib - Failed")

try:
    import yaml
    print("✅ pyyaml - OK")
except:
    print("❌ pyyaml - Failed")

try:
    from tqdm import tqdm
    print("✅ tqdm - OK")
except:
    print("❌ tqdm - Failed")

try:
    import pandas as pd
    print("✅ pandas - OK")
except:
    print("❌ pandas - Failed")

print("\n📦 All imports tested!")