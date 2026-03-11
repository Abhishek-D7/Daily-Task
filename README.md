# AI Document Analyzer

A robust, AI-powered document analysis pipeline that uses open-source Large Language Models (LLMs) via the Hugging Face Inference API. It extracts highly structured data from unstructured text, including summaries, action items, risks, and priority tasks.

The system is designed with strict prompt engineering to prevent hallucinations (returning "Not in context" when data is missing) and includes a resilient multi-model fallback mechanism to ensure high availability.

## Features

* **Strict JSON Extraction:** Forces the LLM to output a clean, parseable JSON schema.
* **Anti-Hallucination:** Grounded prompt design ensures the model only uses explicitly provided text.
* **Model Fallback Routing:** Automatically cycles through a prioritized list of fallback models (like Llama 3.2, Qwen 2.5, and Mistral) if the primary model is busy or unavailable.
* **Dual Interfaces:** Can be run as a lightweight CLI tool or as a modern Web UI with a FastAPI backend.

## Prerequisites

You will need a free Hugging Face API token to run this project. 
Set your token as an environment variable before running the scripts:

**Windows (Command Prompt):** set HF_TOKEN="your_hugging_face_token"
