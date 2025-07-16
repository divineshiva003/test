import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def simulate_user(is_human):
    if is_human:
        return {
            'clicks': np.random.randint(5, 50),
            'avg_typing_interval': np.random.normal(150, 40),
            'std_typing_interval': np.random.normal(20, 10),
            'mouse_path': np.random.normal(7000, 1500),
            'scrolls': np.random.randint(1, 20),
            'pastes': np.random.binomial(1, 0.05),
            'session_duration': np.random.normal(20000, 5000),
            'label': 1
        }
    else:
        return {
            'clicks': np.random.randint(50, 300),
            'avg_typing_interval': np.random.normal(30, 10),
            'std_typing_interval': np.random.normal(5, 3),
            'mouse_path': np.random.normal(1000, 500),
            'scrolls': np.random.randint(0, 3),
            'pastes': np.random.binomial(1, 0.5),
            'session_duration': np.random.normal(5000, 2000),
            'label': 0
        }
data = [simulate_user(True) for _ in range(500)] + [simulate_user(False) for _ in range(500)]
df = pd.DataFrame(data)
df.to_csv("behavior_data.csv", index=False)