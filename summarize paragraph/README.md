# AI Document Analyzer

A robust, AI-powered document analysis pipeline that uses open-source Large Language Models (LLMs) via the Hugging Face Inference API. It extracts highly structured data from unstructured text, including summaries, action items, risks, and priority tasks.

The system is designed with strict prompt engineering to prevent hallucinations (returning "Not in context" when data is missing) and includes a resilient multi-model fallback mechanism to ensure high availability.

## Features

* **Strict JSON Extraction:** Forces the LLM to output a clean, parseable JSON schema.
* **Anti-Hallucination:** Grounded prompt design ensures the model only uses explicitly provided text.
* **Model Fallback Routing:** Automatically cycles through a prioritized list of fallback models (like Llama 3.2, Qwen 2.5, and Mistral) if the primary model is busy or unavailable.
* **Dual Interfaces:** Can be run as a lightweight CLI tool or as a modern Web UI with a FastAPI backend.

## Prerequisites

You will need a free Hugging Face API token to run this project. Set your token as an environment variable before running the scripts:

**Windows (Command Prompt):** export HF_TOKEN="your_hugging_face_token"


DOS
set HF_TOKEN="your_hugging_face_token"

Installation
Clone the repository and install the required dependencies:

Bash
pip install -r requirements.txt

**Usage**
This project offers two distinct ways to run the analysis, depending on your needs.

Method 1: Terminal Execution (main.py)
Use this method for quick, lightweight terminal output. It reads a local text file and prints the parsed JSON directly to your console.

Run the script:
python main.py

Method 2: Web UI + API (api.py & app.py)
Use this method to launch a full-stack microservice architecture. It features a FastAPI backend to process requests and a Streamlit frontend that allows you to paste text or upload .txt files directly from your browser.

You will need two separate terminal windows to run both services simultaneously.

Terminal 1: Start the Backend API
uvicorn api:app --reload

(The API will start running at http://127.0.0.1:8000)

Terminal 2: Start the Frontend UI
streamlit run app.py
