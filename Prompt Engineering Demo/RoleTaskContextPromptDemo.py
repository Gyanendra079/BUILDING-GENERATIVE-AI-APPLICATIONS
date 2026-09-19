# ============================================================
# HUGGING FACE + LANGCHAIN
# Role Context Prompt Example
# ============================================================

# Install:
# python -m pip install -U langchain-core
# python -m pip install -U langchain-huggingface
# python -m pip install -U huggingface_hub
# python -m pip install -U python-dotenv


import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# ------------------------------------------------------------
# 1. Load environment variables
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# 2. Read Hugging Face API token
# ------------------------------------------------------------

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN not found. Please add HF_TOKEN to your .env file."
    )


# ------------------------------------------------------------
# 3. Select Hugging Face model
# ------------------------------------------------------------

model_name = "openai/gpt-oss-120b:fastest"


# ------------------------------------------------------------
# 4. Create Hugging Face LLM
# ------------------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id=model_name,
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_new_tokens=300
)


# ------------------------------------------------------------
# 5. Convert Hugging Face LLM into Chat Model
# ------------------------------------------------------------

chat_model = ChatHuggingFace(
    llm=llm
)


# ------------------------------------------------------------
# 6. Define prompt template
# ------------------------------------------------------------

role_context_template = """
You are a highly professional customer support specialist
at a premium tech company.

Your goal is to address the customer's concerns while
maintaining a calm and helpful demeanor.

Customer Message:
{customer_message}

Please provide a response that:

1. Acknowledges the customer's feelings
2. Addresses their concerns professionally
3. Offers concrete solutions
4. Ends with a positive note
"""


role_context_prompt = PromptTemplate(
    input_variables=["customer_message"],
    template=role_context_template
)


# ------------------------------------------------------------
# 7. Customer message
# ------------------------------------------------------------

customer_message = """
I've been waiting for 2 hours for your delivery and it still hasn't arrived!
This is absolutely unacceptable! I paid extra for express delivery and this
is what I get? I want my money back immediately!
"""


# ------------------------------------------------------------
# 8. Create final prompt
# ------------------------------------------------------------

final_prompt = role_context_prompt.invoke(
    {
        "customer_message": customer_message
    }
)


# ------------------------------------------------------------
# 9. Send prompt to Hugging Face model
# ------------------------------------------------------------

response = chat_model.invoke(final_prompt)


# ------------------------------------------------------------
# 10. Print response
# ------------------------------------------------------------

print("\n========== AI RESPONSE ==========\n")

print(response.content)