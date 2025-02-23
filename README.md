# Canary Analysis Bot

Canary Analysis Bot is a tool designed to analyze and interpret canary deployment data for multiple services. It aggregates performance metrics, error rates, latencies, resource usage, and throughput, then provides detailed insights into the differences between baseline and canary deployments. The bot leverages a trained model to generate human-readable explanations of the data and features a conversational interface built with Gradio.

## Features

- **Comprehensive Data Analysis:**  
  Aggregates key metrics such as baseline and canary values, error rates, latencies, CPU and memory usage, and throughput.

- **Multiple Query Categories:**  
  Supports queries for summarizing data, comparing metrics, detecting anomalies, analyzing trends, comparing environments, and generating detailed reports.

- **Natural Language Interpretation:**  
  Uses a GPT-based model to translate raw statistics into clear, expert-style interpretations.

- **Chat-like Interface:**  
  Interact with the bot through a conversational web UI built with Gradio. Users can submit queries by pressing Enter or clicking a "Send" button.

- **Threaded Processing:**  
  Each query is processed in its own thread for improved responsiveness.

## Project Structure

```
CanaryScope/
├── data/
│   └── sample_canary_data.csv    # Generated sample data file
├── src/
│   ├── canary_data_utils.py      # Data loading and analysis functions
│   └── canary_nl_bot.py          # Natural language chat interface with Gradio
├── .env                          # Environment file containing API key
└── README.md
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/CanaryScope.git
   cd CanaryScope
   ```

2. **Install Dependencies:**
   Use pip to install the required Python packages:
   ```bash
   pip install openai python-dotenv pandas numpy gradio
   ```

3. **Set Up Environment Variables:**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_actual_api_key_here
   ```

4. **Generate Sample Data:**
   If you need sample data, run the data generator script:
   ```bash
   python generate_complex_sample_data.py
   ```

## Usage

### Running the Chat Interface

Start the chat interface by executing:
```bash
python3 src/canary_nl_bot.py
```

This command will launch a local Gradio web server. Open the provided URL (e.g., `http://127.0.0.1:7860/`) in your browser to interact with the bot.

### Example Queries

Try queries such as:

- **Summarize:** "Provide a summary for the auth service."
- **Compare:** "Compare metrics for the billing service."
- **Anomaly Detection:** "Detect anomalies in baseline_latency for search."
- **Trend Analysis:** "Analyze the trend of canary_value for billing in production."
- **Environment Comparison:** "Compare baseline_memory for the recommendation service between production and staging."
- **Detailed Report:** "Give me a detailed report for billing."

The bot will return both raw numerical data and an expert-style explanation interpreting the results.

## Customization

### Extending Analysis
You can extend the functions in `canary_data_utils.py` to incorporate additional metrics or more advanced statistical methods. Common extensions include:

- Adding new statistical measures
- Implementing custom anomaly detection algorithms
- Creating specialized visualizations
- Adding support for new data sources

### Modifying Query Interpretation
The GPT-based interpretation in `canary_nl_bot.py` is driven by a system prompt that can be adjusted to better suit your domain. Consider customizing:

- The language style and tone
- Domain-specific terminology
- The level of technical detail in responses
- The types of insights provided

### UI Customization
The Gradio interface is built using Blocks and is highly customizable. Some common modifications include:

- Adding new input components
- Customizing the chat layout
- Implementing additional visualization options
- Adding authentication or access controls

## Advanced Features

### Data Processing Pipeline

The bot implements a sophisticated data processing pipeline that includes:

1. **Data Validation:**
   - Schema validation
   - Data type checking
   - Missing value handling
   - Outlier detection

2. **Metric Aggregation:**
   - Time-based aggregation
   - Service-level grouping
   - Environment-specific analysis
   - Statistical summarization

3. **Analysis Capabilities:**
   - Trend detection
   - Anomaly identification
   - Performance comparison
   - Resource utilization analysis

### Error Handling

The system includes robust error handling for common scenarios:

- Invalid query formats
- Missing or corrupted data
- API connection issues
- Resource constraints
- Authentication failures


## Contributing

Contributions are welcome! Feel free to make a PR!