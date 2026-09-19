# python -m pip install -U langchain
# python -m pip install -U langchain-core langchain-huggingface huggingface_hub python-dotenv

# ============================================================
# HUGGING FACE API + PROMPT TEMPLATE
# Customer Complaint Response Generator
# ============================================================

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate


# ------------------------------------------------------------
# 1. Load environment variables
# ------------------------------------------------------------

load_dotenv()

# Read Hugging Face API key from .env
hf_api_key = os.getenv("HF_TOKEN")

if not hf_api_key:
    raise ValueError(
        "HF_TOKEN not found. Please add your Hugging Face API key "
        "to the .env file."
    )


# ------------------------------------------------------------
# 2. Initialize Hugging Face client
# ------------------------------------------------------------

client = InferenceClient(
    api_key=hf_api_key
)


# ------------------------------------------------------------
# 3. Select Hugging Face model
# ------------------------------------------------------------

model_name ="openai/gpt-oss-120b:fastest"


# ------------------------------------------------------------
# 4. Define prompt template
# ------------------------------------------------------------

basic_template = """
You are a professional customer support representative.

Reply politely and professionally to the following customer complaint.

Customer message:
{customer_message}

Your response should:
- Apologize for the inconvenience
- Acknowledge the customer's frustration
- Explain that the issue will be addressed
- Offer an appropriate next step
- Remain professional and empathetic
"""

basic_prompt = PromptTemplate(
    input_variables=["customer_message"],
    template=basic_template
)


# ------------------------------------------------------------
# 5. Customer message
# ------------------------------------------------------------

customer_message = """
I've been waiting for 2 hours for your delivery and it still hasn't arrived!
This is absolutely unacceptable! I paid extra for express delivery and this
is what I get? I want my money back immediately!
"""


# ------------------------------------------------------------
# 6. Create final prompt
# ------------------------------------------------------------

final_prompt = basic_prompt.format(
    customer_message=customer_message
)


# ------------------------------------------------------------
# 7. Send request to Hugging Face
# ------------------------------------------------------------

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": final_prompt
        }
    ],
    temperature=0.7,
    max_tokens=300
)


# ------------------------------------------------------------
# 8. Print response
# ------------------------------------------------------------

print("\n========== AI RESPONSE ==========\n")

print(response.choices[0].message.content)




