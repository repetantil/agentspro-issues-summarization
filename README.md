# GitHub Issues Summarizer

A Python script that uses LangChain, LangGraph, and MCP (Model Context Protocol) to fetch, filter, and summarize GitHub issues using an LLM.

## Features

- Fetches issues from any GitHub repository using MCP
- Filters issues by topic using keyword matching
- Generates AI-powered summaries using OpenAI's GPT models
- Multi-step orchestration with LangGraph
- Simple CLI interface

## Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GITHUB_TOKEN="your-github-personal-access-token"
   
   # Optional: Enable LangSmith tracing
   export LANGCHAIN_TRACING_V2=true # Enable LangSmith tracing
   export LANGCHAIN_API_KEY=ls__your-key       # Get from https://smith.langchain.com
   export LANGCHAIN_PROJECT=pr-back-transportation-56
   ```

## Usage

Run the script with default parameters (nvm-sh/nvm repo, "build issues" topic):
```bash
python issues_summarizer.py
```

Or specify custom repository and topic:
```bash
python issues_summarizer.py --repo microsoft/vscode --topic "copilot"
```

## How It Works

The script uses LangGraph to orchestrate three nodes:

1. **Fetch Issues Node**: Uses MCP GitHub client to fetch the last 100 issues from the repository
2. **Filter Issues Node**: Filters issues based on keyword matching with the topic
3. **Summarize Node**: Uses OpenAI's GPT to analyze and summarize the filtered issues

The summary includes:
- Total number of matching issues
- Open vs closed counts
- Common themes and patterns
- Typical error messages
- Common resolutions and workarounds
- Notable trends

## LangSmith Integration

This script includes optional LangSmith integration for tracing and monitoring:

- **Automatic tracing** of all LangChain/LangGraph operations
- **Visual workflow visualization** in LangSmith dashboard
- **Performance monitoring** and debugging
- **Token usage tracking**

To enable LangSmith:
1. Sign up for a free account at [smith.langchain.com](https://smith.langchain.com)
2. Get your API key from the settings
3. Set the `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` environment variables
4. Run the script - traces will appear in your LangSmith project

The script will work without LangSmith; it's completely optional.

## Requirements

- Python 3.10+
- OpenAI API key
- GitHub Personal Access Token
- Node.js (for MCP GitHub server via npx)
- LangSmith API key (optional, for tracing)