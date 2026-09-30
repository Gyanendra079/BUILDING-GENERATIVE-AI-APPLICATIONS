# python -m pip install -U langchain-classic langchain-openai openai python-dotenv requests
# python -m pip install -U langchain-openai openai

# ============================================================
# CONVERSATION SUMMARY MEMORY
# HUGGING FACE CURRENT PROVIDER MODEL
# ============================================================

import os
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryMemory


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN was not found in .env"
    )


# ============================================================
# 2. GET CURRENT HUGGING FACE MODELS
# ============================================================

models_url = "https://router.huggingface.co/v1/models"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

response = requests.get(
    models_url,
    headers=headers,
    timeout=30
)

if response.status_code != 200:

    raise RuntimeError(
        f"Hugging Face model list failed.\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}"
    )


models = response.json().get("data", [])


if not models:

    raise RuntimeError(
        "No chat models are currently available "
        "through your Hugging Face account."
    )


# ============================================================
# 3. SELECT FIRST AVAILABLE MODEL
# ============================================================

model_id = models[0]["id"]


print()
print("=" * 70)
print("HUGGING FACE MODEL SELECTED")
print("=" * 70)

print("Model:", model_id)


# ============================================================
# 4. CREATE HUGGING FACE CHAT MODEL
# ============================================================

llm = ChatOpenAI(
    model=model_id,
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.1,
    max_tokens=256
)


# ============================================================
# 5. CREATE SCHEDULE
# ============================================================

schedule = """
There is a meeting at 8am with your product team.
You will need your PowerPoint presentation prepared.

9am-12pm have time to work on your LangChain project.

At Noon, lunch at the Italian restaurant with a customer
who is driving from over an hour away to meet you
to understand the latest in AI.

Be sure to bring your laptop to show the latest LLM demo.
"""


# ============================================================
# 6. CREATE SUMMARY MEMORY
# ============================================================

memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=100
)


# ============================================================
# 7. SAVE CONVERSATION
# ============================================================

memory.save_context(
    {"input": "Hello"},
    {"output": "What's up"}
)

memory.save_context(
    {"input": "Not much, just hanging"},
    {"output": "Cool"}
)

memory.save_context(
    {"input": "What is on the schedule today?"},
    {"output": schedule}
)


# ============================================================
# 8. DISPLAY SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONVERSATION SUMMARY")
print("=" * 70)

print(
    memory.load_memory_variables({})
)


# ============================================================
# 9. CREATE CONVERSATION
# ============================================================

conversation = ConversationChain(
    memory=memory,
    llm=llm,
    verbose=True
)


# ============================================================
# 10. QUESTIONS
# ============================================================

print()
print("=" * 70)
print("QUESTION 1")
print("=" * 70)

print(
    conversation.predict(
        input="9am-12pm what will you do?"
    )
)


print()
print("=" * 70)
print("QUESTION 2")
print("=" * 70)

print(
    conversation.predict(
        input="What is deep learning?"
    )
)


print()
print("=" * 70)
print("QUESTION 3")
print("=" * 70)

print(
    conversation.predict(
        input="What is AI?"
    )
)