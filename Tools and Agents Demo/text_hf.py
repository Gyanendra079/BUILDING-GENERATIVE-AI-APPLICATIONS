import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ------------------------------------------------------------
# Load .env
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError(
        f"API_KEY not found.\n"
        f"Expected .env at:\n{ENV_FILE}"
    )


# ------------------------------------------------------------
# Hugging Face OpenAI-compatible client
# ------------------------------------------------------------

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=API_KEY
)


# ------------------------------------------------------------
# Test model
# ------------------------------------------------------------

response = client.chat.completions.create(
    model="openai/gpt-oss-120b:cerebras",
    messages=[
        {
            "role": "user",
            "content": "Explain Natural Language Processing in two sentences."
        }
    ]
)


# ------------------------------------------------------------
# Display response
# ------------------------------------------------------------

print("=" * 70)
print("HUGGING FACE TEST")
print("=" * 70)

print(response.choices[0].message.content)

print("=" * 70)
