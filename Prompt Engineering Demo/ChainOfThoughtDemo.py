# ============================================================
# CUSTOMER SUPPORT CHAIN USING HUGGING FACE INFERENCE API
# ============================================================

# Install:
# pip install -U huggingface_hub python-dotenv langchain langchain-core


import os
from dotenv import load_dotenv

from huggingface_hub import InferenceClient

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN not found.\n"
        "Please create a .env file and add:\n"
        "HF_TOKEN=hf_your_token_here"
    )


# ============================================================
# 2. CREATE HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=hf_token,
    provider="auto"
)


# ============================================================
# 3. SELECT MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# 4. CREATE PROMPT
# ============================================================

chain_of_thought_template = """
You are a highly professional customer support specialist
at a premium technology company.

Your goal is to address the customer's concerns while
maintaining a calm, empathetic and helpful demeanor.

Analyze the customer message using the following format.

1. Sentiment:
Identify the customer's emotional state.

2. Core Issues:
Identify the main problems mentioned by the customer.

3. Action Plan:
Briefly explain what the support agent should do.

4. Final Response:
Write a professional response to the customer.

IMPORTANT:
Do not reveal private or hidden chain-of-thought.
Provide only a concise reasoning summary.

Customer Message:

{customer_message}

Now provide the analysis and final response.
"""


# ============================================================
# 5. CREATE LANGCHAIN PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["customer_message"],
    template=chain_of_thought_template
)


# ============================================================
# 6. FUNCTION TO CALL HUGGING FACE
# ============================================================

def call_huggingface(prompt_text):

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],

        temperature=0.7,
        max_tokens=600
    )

    return response.choices[0].message.content


# ============================================================
# 7. CREATE LANGCHAIN CHAIN
# ============================================================

chain = prompt | RunnableLambda(
    lambda x: call_huggingface(x.to_string())
)


# ============================================================
# 8. CUSTOMER MESSAGE
# ============================================================

customer_message = """
I've been waiting for 2 hours for your delivery and it still
hasn't arrived!

This is absolutely unacceptable!

I paid extra for express delivery and this is what I get?

I want my money back immediately.
"""


# ============================================================
# 9. RUN THE CHAIN
# ============================================================

response = chain.invoke(
    {
        "customer_message": customer_message
    }
)


# ============================================================
# 10. DISPLAY RESULT WITH MARKDOWN FORMATTING
# ============================================================

from rich.console import Console
from rich.markdown import Markdown

console = Console()

print("\n")
print("=" * 70)
print("HUGGING FACE CUSTOMER SUPPORT CHAIN")
print("=" * 70)

console.print(Markdown(response))

print("=" * 70)

