# 🛠️ AI Agents & Tool Calling — Project Implementation

This repository contains a collection of Python projects demonstrating **LLM tools, tool calling, AI agents, and multi-agent workflows**, developed and implemented by **Gyanendra**.

## 🎯 Project Overview

This project showcases how AI agents can go beyond generating text by **using tools to interact with external systems and perform real-world tasks**. The repository serves as a practical implementation guide for building intelligent, autonomous systems.

Key implementations include:
* Integrating external tools (Calculators, Search, etc.) with Large Language Models.
* Building tool-calling workflows for autonomous decision-making.
* Connecting AI agents directly to SQL databases for natural language querying.
* Performing automated web scraping using AI-powered workflows.
* Architecting multi-agent systems using CrewAI for complex, multi-step business use cases.

## 🧠 Core Architecture: Tools vs. AI Agents

In this project, a **tool** gives an LLM access to a specific capability (e.g., a Calculator, Web scraper, SQL database, or API). 

An **AI agent** is the reasoning engine that decides **which tool to use, when to use it, and how to combine tools to accomplish a user's task**.

### Standard Agent Workflow Implemented
```
User Request
     ↓
     LLM (Reasoning)
     ↓
Decide What To Do
     ↓
Select Tool
     ↓
Execute Tool
     ↓
Observe Result
     ↓
Final Answer Synthesis
```

## 📂 Project Modules & Details

| Module | Description | 
 | ----- | ----- | 
| [`MathTools_Agents_Usecase.py`](./MathTools_Agents_Usecase.py) | Demonstrates an AI agent routing calculation tasks to mathematical tools rather than relying on LLM hallucination-prone math capabilities. | 
| [`SQLDatabaseTool_SqlAgent.py`](./SQLDatabaseTool_SqlAgent.py) | An agent interacting with a SQL database, converting natural language into SQL queries, executing them, and summarizing the results. | 
| [`WebScraping_Using_Apify.py`](./WebScraping_Using_Apify.py) | Implementation of web scraping using the Apify API to feed live internet data into the AI application. | 
| [`CrewAI_Usecase1_WebScraping.py`](./CrewAI_Usecase1_WebScraping.py) | A multi-agent CrewAI implementation for coordinated web scraping and data extraction. | 
| [`CrewAI_Usecase2_Personalized_Email_Drafts.py`](./CrewAI_Usecase2_Personalized_Email_Drafts.py) | Uses specialized CrewAI agents to research targets and draft highly personalized outreach emails. | 
| [`CrewAI_Usecase3_Trading_Platform.py`](./CrewAI_Usecase3_Trading_Platform.py) | A complex multi-agent simulation of a trading platform, dividing roles among risk, analysis, and execution agents. *(Note: For educational/demonstration purposes only).* | 

## 🏗️ System Evolution

The scripts in this project are designed to demonstrate the architectural evolution of AI systems:
`Basic Tools → Tool Calling → Single AI Agent → Agent + External Systems → Multi-Agent System`

## ⚙️ Prerequisites & Setup

To run this project locally, you will need:
* Python 3.9+
* OpenAI API key
* Apify account/API key (for scraping modules)
* A sample SQL database (for the SQL agent module)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME/07-tools-and-agents-demo
```

2. Install the required dependencies:
```bash
# Core dependencies
pip install langchain langchain-openai python-dotenv

# For CrewAI modules
pip install crewai

# For Apify scraping modules
pip install apify-client
```

## 🔐 Environment Configuration

This project requires secure management of API keys. Never hard-code your keys into the Python scripts. 

Create a `.env` file in the root directory and input your credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
APIFY_API_TOKEN=your_apify_api_token_here
```

The scripts utilize the `python-dotenv` package to securely load these variables at runtime. Ensure your `.env` file is added to your `.gitignore` to prevent accidental exposure.

## 📌 Implementation Highlights

By reviewing and running the modules in this project, you will see practical code examples of:
* **Tool Binding:** How to wrap Python functions into a format the LLM can invoke.
* **Agent Initialization:** Configuring zero-shot React agents in Langchain.
* **Role-Based Delegation:** Setting up specialized personas in CrewAI to break down complex tasks.