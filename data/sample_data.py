# generate_complex_sample_data.py

import pandas as pd
import numpy as np

# Parameters
num_points = 720  # hourly data for one month
services = ['auth', 'billing', 'search', 'recommendation']
environments = ['production', 'staging']

data = []

for service in services:
    for env in environments:
        # Generate a range of timestamps
        timestamps = pd.date_range(start='2023-01-01', periods=num_points, freq='H')
        
        # Simulate metrics using random data
        baseline_value = np.random.normal(loc=100, scale=15, size=num_points)
        canary_value = baseline_value + np.random.normal(loc=0, scale=10, size=num_points)
        baseline_error_rate = np.abs(np.random.normal(loc=0.03, scale=0.01, size=num_points))
        canary_error_rate = baseline_error_rate + np.random.normal(loc=0.005, scale=0.005, size=num_points)
        baseline_latency = np.random.normal(loc=200, scale=30, size=num_points)
        canary_latency = baseline_latency + np.random.normal(loc=5, scale=15, size=num_points)
        baseline_cpu = np.random.normal(loc=50, scale=10, size=num_points)
        canary_cpu = baseline_cpu + np.random.normal(loc=0, scale=5, size=num_points)
        baseline_memory = np.random.normal(loc=1024, scale=100, size=num_points)
        canary_memory = baseline_memory + np.random.normal(loc=0, scale=50, size=num_points)
        baseline_throughput = np.random.normal(loc=500, scale=50, size=num_points)
        canary_throughput = baseline_throughput + np.random.normal(loc=10, scale=20, size=num_points)
        
        # Append a row for each timestamp
        for i in range(num_points):
            data.append({
                'timestamp': timestamps[i],
                'service': service,
                'environment': env,
                'baseline_value': baseline_value[i],
                'canary_value': canary_value[i],
                'baseline_error_rate': baseline_error_rate[i],
                'canary_error_rate': canary_error_rate[i],
                'baseline_latency': baseline_latency[i],
                'canary_latency': canary_latency[i],
                'baseline_cpu': baseline_cpu[i],
                'canary_cpu': canary_cpu[i],
                'baseline_memory': baseline_memory[i],
                'canary_memory': canary_memory[i],
                'baseline_throughput': baseline_throughput[i],
                'canary_throughput': canary_throughput[i]
            })

# Create a DataFrame and write to CSV
df = pd.DataFrame(data)
df.to_csv('sample_canary_data.csv', index=False)
print("New, more complex sample canary data saved to sample_canary_data.csv")
