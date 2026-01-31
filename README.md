# LLM Agent for ML Prediction using Groq
An agentic machine learning inference system that integrates a pre-trained ML model with a tool-calling LLM agent powered by Groq and LangChain.
The agent understands natural language queries, extracts numerical features, invokes an ML prediction tool, and returns both the prediction and an explanation.

# Project Overview
Traditional ML models require structured inputs and manual execution.
This project wraps an ML prediction pipeline inside an LLM-driven agent, enabling:
* Natural language interaction <br>
* Automatic tool invocation <br>
* Seamless ML inference through an AI agent <br>
This system demonstrates how agentic AI can orchestrate classical ML models in real-world applications.

# System workflow
* User provides a natural language query with feature values <br>
* Groq-powered LLM interprets the intent <br>
* LangChain agent selects the appropriate tool <br>
* ML prediction tool: <br>
   * Scales features using a saved StandardScaler <br>
   * Predicts output using a trained XGBoost model <br>
* Agent returns the prediction with contextual explanation <br>

User Query <br>
   ↓ <br>
Groq LLM (ChatGroq) <br>
   ↓ <br>
LangChain Tool-Calling Agent <br>
   ↓ <br>
ML Prediction Tool <br>
   ↓ <br>
Prediction + Explanation <br>

# Tech Stack
### Language: Python
### LLM Provider: Groq
### Model: Qwen (via Groq API)
### Agent Framework: LangChain
### ML Model: XGBoost
### Preprocessing: StandardScaler (scikit-learn)
### Model Persistence: joblib


