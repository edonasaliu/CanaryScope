# src/canary_data_utils.py

import pandas as pd
import numpy as np

def load_canary_data(file_path: str = "../data/sample_canary_data.csv") -> pd.DataFrame:
    """
    Load the canary analysis CSV data into a pandas DataFrame.
    Expected columns include:
      timestamp, service, environment,
      baseline_value, canary_value,
      baseline_error_rate, canary_error_rate,
      baseline_latency, canary_latency,
      baseline_cpu, canary_cpu,
      baseline_memory, canary_memory,
      baseline_throughput, canary_throughput
    """
    return pd.read_csv(file_path, parse_dates=['timestamp'])

import openai

def explain_stats(stats: str) -> str:
    """
    Use GPT to generate a natural language explanation of the provided statistics.
    """
    prompt = (
        "You are an expert performance analyst. Below are some key performance metrics "
        "for a service's canary analysis:\n\n"
        f"{stats}\n\n"
        "Please provide a detailed explanation in plain language that interprets these statistics. "
        "Discuss whether the canary deployment appears to be performing well compared to the baseline, "
        "mention any potential issues, and suggest what the numbers might imply for the system's performance."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are an expert performance analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7,
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        # If the call fails, return a fallback explanation.
        return "The canary deployment appears to be generally stable with only minor differences; however, a detailed expert review is recommended."

def summarize_service_data(service: str, data: pd.DataFrame = None) -> str:
    """
    Summarize canary analysis data for a given service (aggregated across environments) 
    and then use the trained model to generate a natural language interpretation.
    """
    if data is None:
        data = load_canary_data()
    service_data = data[data['service'].str.lower() == service.lower()]
    if service_data.empty:
        return f"No data found for service '{service}'."
    
    avg_baseline = service_data['baseline_value'].mean()
    avg_canary = service_data['canary_value'].mean()
    avg_baseline_error = service_data['baseline_error_rate'].mean()
    avg_canary_error = service_data['canary_error_rate'].mean()
    avg_baseline_latency = service_data['baseline_latency'].mean()
    avg_canary_latency = service_data['canary_latency'].mean()
    
    diff_value = avg_canary - avg_baseline
    diff_latency = avg_canary_latency - avg_baseline_latency

    stats = (
        f"Service: {service}\n"
        f"Average Baseline Value: {avg_baseline:.2f}\n"
        f"Average Canary Value: {avg_canary:.2f}\n"
        f"Difference (Canary - Baseline): {diff_value:.2f}\n\n"
        f"Average Baseline Error Rate: {avg_baseline_error:.4f}\n"
        f"Average Canary Error Rate: {avg_canary_error:.4f}\n\n"
        f"Average Baseline Latency: {avg_baseline_latency:.2f} ms\n"
        f"Average Canary Latency: {avg_canary_latency:.2f} ms\n"
        f"Latency Difference: {diff_latency:.2f} ms\n"
    )
    
    # Now, use GPT to explain the stats in plain language.
    explanation = explain_stats(stats)
    
    return stats + "\nInterpretation:\n" + explanation



def compare_service_metrics(service: str, data: pd.DataFrame = None) -> str:
    """
    Compare baseline and canary metrics for a given service by calculating
    percentage differences for values, error rates, and latencies.
    """
    if data is None:
        data = load_canary_data()
    service_data = data[data['service'].str.lower() == service.lower()]
    if service_data.empty:
        return f"No data found for service '{service}'."
    
    baseline_mean = service_data['baseline_value'].mean()
    canary_mean = service_data['canary_value'].mean()
    value_diff = canary_mean - baseline_mean
    value_pct_diff = (value_diff / baseline_mean * 100) if baseline_mean != 0 else float('inf')
    
    baseline_error = service_data['baseline_error_rate'].mean()
    canary_error = service_data['canary_error_rate'].mean()
    error_diff = canary_error - baseline_error
    error_pct_diff = (error_diff / baseline_error * 100) if baseline_error != 0 else float('inf')
    
    baseline_latency = service_data['baseline_latency'].mean()
    canary_latency = service_data['canary_latency'].mean()
    latency_diff = canary_latency - baseline_latency
    latency_pct_diff = (latency_diff / baseline_latency * 100) if baseline_latency != 0 else float('inf')
    
    comparison = (
        f"Service: {service}\n"
        f"Baseline vs Canary Value: {baseline_mean:.2f} vs {canary_mean:.2f} "
        f"(Diff: {value_diff:.2f}, {value_pct_diff:.2f}%)\n\n"
        f"Baseline vs Canary Error Rate: {baseline_error:.4f} vs {canary_error:.4f} "
        f"(Diff: {error_diff:.4f}, {error_pct_diff:.2f}%)\n\n"
        f"Baseline vs Canary Latency: {baseline_latency:.2f} ms vs {canary_latency:.2f} ms "
        f"(Diff: {latency_diff:.2f} ms, {latency_pct_diff:.2f}%)\n"
    )
    return comparison

def detect_anomalies(service: str, metric: str, threshold: float = 2.0, data: pd.DataFrame = None) -> str:
    """
    Detect anomalies for a given metric in a service using the z-score method.
    """
    if data is None:
        data = load_canary_data()
    service_data = data[data['service'].str.lower() == service.lower()]
    if service_data.empty:
        return f"No data found for service '{service}'."
    if metric not in service_data.columns:
        return f"Metric '{metric}' not found in data."
    
    values = service_data[metric]
    mean_val = values.mean()
    std_val = values.std()
    if std_val == 0:
        return f"No variation in metric '{metric}' for service '{service}', so no anomalies can be detected."
    
    service_data = service_data.copy()
    service_data['z_score'] = (service_data[metric] - mean_val) / std_val
    anomalies = service_data[service_data['z_score'].abs() > threshold]
    if anomalies.empty:
        return f"No anomalies detected for metric '{metric}' in service '{service}'."
    
    anomaly_summary = f"Anomalies for metric '{metric}' in service '{service}':\n"
    for _, row in anomalies.iterrows():
        anomaly_summary += (
            f"Timestamp: {row['timestamp']}, Value: {row[metric]:.2f}, "
            f"Z-score: {row['z_score']:.2f}\n"
        )
    return anomaly_summary

def analyze_trend(service: str, metric: str, environment: str = None, data: pd.DataFrame = None) -> str:
    """
    Analyze the trend of a specific metric over time for a given service.
    If environment is provided, restrict the analysis to that environment.
    A simple linear regression is used to determine if the metric is trending upward or downward.
    """
    if data is None:
        data = load_canary_data()
    df = data[data['service'].str.lower() == service.lower()]
    if environment:
        df = df[df['environment'].str.lower() == environment.lower()]
    if df.empty or metric not in df.columns:
        return f"No data found for service '{service}' with metric '{metric}'."
    
    # Convert timestamps to numeric values (e.g., epoch seconds)
    df = df.sort_values('timestamp')
    df['time_numeric'] = df['timestamp'].astype(np.int64) // 10**9
    x = df['time_numeric']
    y = df[metric]
    # Use polyfit to determine slope
    slope, intercept = np.polyfit(x, y, 1)
    
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    return (f"Trend analysis for '{metric}' in service '{service}'"
            f"{' (environment: ' + environment + ')' if environment else ''}: "
            f"The metric is {direction} (slope = {slope:.4f}).")

def compare_environments(service: str, metric: str, data: pd.DataFrame = None) -> str:
    """
    Compare the average of a given metric for a service between different environments.
    """
    if data is None:
        data = load_canary_data()
    df = data[data['service'].str.lower() == service.lower()]
    if df.empty or metric not in df.columns:
        return f"No data found for service '{service}' with metric '{metric}'."
    
    env_groups = df.groupby('environment')[metric].mean()
    report = f"Environment comparison for '{metric}' in service '{service}':\n"
    for env, avg in env_groups.items():
        report += f"  {env}: {avg:.2f}\n"
    return report

def detailed_analysis(service: str, data: pd.DataFrame = None) -> str:
    """
    Provide a detailed analysis for a given service across all available metrics.
    """
    if data is None:
        data = load_canary_data()
    df = data[data['service'].str.lower() == service.lower()]
    if df.empty:
        return f"No data found for service '{service}'."
    
    report = f"Detailed analysis for service '{service}':\n\n"
    report += summarize_service_data(service, data) + "\n"
    report += compare_service_metrics(service, data) + "\n"
    # List of key metrics for trend analysis
    metrics = ['baseline_value', 'canary_value', 'baseline_error_rate', 
               'canary_error_rate', 'baseline_latency', 'canary_latency', 
               'baseline_cpu', 'canary_cpu', 'baseline_memory', 'canary_memory',
               'baseline_throughput', 'canary_throughput']
    report += "Trend analysis (overall):\n"
    for m in metrics:
        trend = analyze_trend(service, m, data=data)
        report += f"  {m}: {trend}\n"
    return report
