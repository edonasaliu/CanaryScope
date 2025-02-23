# src/canary_nl_bot.py

import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

import logging
import re
import threading
import gradio as gr
from canary_data_utils import (
    load_canary_data,
    summarize_service_data,
    compare_service_metrics,
    detect_anomalies,
    analyze_trend,
    compare_environments,
    detailed_analysis
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the complex canary analysis data
data = load_canary_data()

def interpret_query(query: str) -> dict:
    """
    Use GPT to interpret the user's natural language query and determine:
      - action: one of 'summarize', 'compare', 'anomaly', 'trend', 'env_compare', or 'detailed'
      - service: the target service name
      - metric (optional): the metric to focus on.
      - environment (optional): for trend or environment comparison queries.
    
    If GPT fails, a fallback regex parser is used.
    """
    system_prompt = (
        "You are an assistant that extracts instructions for canary analysis queries. "
        "Given a natural language query, output a JSON object with the keys: "
        "\"action\", \"service\", and optionally \"metric\" and \"environment\". "
        "The action can be one of \"summarize\", \"compare\", \"anomaly\", \"trend\", \"env_compare\", or \"detailed\". "
        "For example, if the query is \"Provide a summary for the auth service\", "
        "output: {\"action\": \"summarize\", \"service\": \"auth\"}. "
        "If the query is \"Detect anomalies in baseline_latency for the search service\", "
        "output: {\"action\": \"anomaly\", \"service\": \"search\", \"metric\": \"baseline_latency\"}. "
        "If the query is \"Analyze the trend of canary_value for billing in production\", "
        "output: {\"action\": \"trend\", \"service\": \"billing\", \"metric\": \"canary_value\", \"environment\": \"production\"}. "
        "If the metric is not specified for an anomaly query, default to \"baseline_value\"."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=150,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()
        result = json.loads(reply)
        return result
    except Exception as e:
        logging.error(f"Error interpreting query with GPT: {e}")
        logging.info("Falling back to basic regex parsing.")
        lower_query = query.lower()
        if "summarize" in lower_query:
            action = "summarize"
        elif "compare" in lower_query:
            action = "compare"
        elif "anomaly" in lower_query or "detect" in lower_query:
            action = "anomaly"
        elif "trend" in lower_query or "analyze trend" in lower_query:
            action = "trend"
        elif "env" in lower_query or "environment" in lower_query:
            action = "env_compare"
        elif "detailed" in lower_query or "full report" in lower_query:
            action = "detailed"
        else:
            action = None

        service_match = re.search(r'(\w+)\s+service', query, re.IGNORECASE)
        service = service_match.group(1) if service_match else None
        metric_match = re.search(
            r'(baseline_value|canary_value|baseline_error_rate|canary_error_rate|'
            r'baseline_latency|canary_latency|baseline_cpu|canary_cpu|'
            r'baseline_memory|canary_memory|baseline_throughput|canary_throughput)',
            query, re.IGNORECASE)
        metric = metric_match.group(1) if metric_match else "baseline_value"
        env_match = re.search(r'(production|staging)', query, re.IGNORECASE)
        environment = env_match.group(1) if env_match else None

        return {"action": action, "service": service, "metric": metric, "environment": environment}

def query_bot(user_query: str) -> str:
    """
    Process the query by interpreting it and then calling the appropriate data analysis function.
    """
    interpretation = interpret_query(user_query)
    logging.info(f"Interpreted query: {interpretation}")

    action = interpretation.get("action")
    service = interpretation.get("service")
    metric = interpretation.get("metric", "baseline_value")
    environment = interpretation.get("environment")

    if not action or not service:
        return "Could not properly interpret your query. Please try rephrasing."

    if action == "summarize":
        return summarize_service_data(service, data)
    elif action == "compare":
        return compare_service_metrics(service, data)
    elif action == "anomaly":
        return detect_anomalies(service, metric, data=data)
    elif action == "trend":
        return analyze_trend(service, metric, environment=environment, data=data)
    elif action == "env_compare":
        return compare_environments(service, metric, data=data)
    elif action == "detailed":
        return detailed_analysis(service, data=data)
    else:
        return "Unknown action requested."

# --- Threaded Execution Section ---
def process_query(query: str, result_container: list):
    """
    Thread target that calls query_bot and appends the result.
    """
    result = query_bot(query)
    result_container.append(result)

def query_bot_threaded(user_query: str) -> str:
    """
    Launch a thread to process the query.
    """
    result_container = []
    thread = threading.Thread(target=process_query, args=(user_query, result_container))
    thread.start()
    thread.join()  # Wait for the thread to finish
    return result_container[0] if result_container else "No result returned."

# --- Chat-like Gradio Interface ---
def chat(user_message, chat_history):
    """
    Update the conversation with the user's message and the bot's response.
    """
    chat_history = chat_history or []
    # Append user message (bot response is None initially)
    chat_history.append((user_message, None))
    # Process the user's message in a separate thread
    bot_response = query_bot_threaded(user_message)
    # Update the last conversation entry with the bot's response
    chat_history[-1] = (user_message, bot_response)
    # Return three outputs: reset input, updated chat history for chatbot, and updated state.
    return "", chat_history, chat_history

def main():
    print("Natural Language Canary Analysis Chat Bot (Gradio Web UI)")
    with gr.Blocks() as demo:
        gr.Markdown("## Canary Analysis Bot Chat")
        chatbot = gr.Chatbot()
        state = gr.State([])
        user_input = gr.Textbox(lines=2, placeholder="Enter your message here...", label="Your Message")
        send_button = gr.Button("Send")
        send_button.click(chat, inputs=[user_input, state], outputs=[user_input, chatbot, state])
        user_input.submit(chat, inputs=[user_input, state], outputs=[user_input, chatbot, state])
    demo.launch()

if __name__ == "__main__":
    main()
