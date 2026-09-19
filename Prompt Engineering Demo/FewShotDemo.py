# ============================================================
# FEW-SHOT PROMPTING USING HUGGING FACE + LANGCHAIN
# ============================================================

# Install required packages:
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
# 2. Read Hugging Face API key
# ------------------------------------------------------------

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN not found. Please add your Hugging Face API key "
        "to the .env file."
    )


# ------------------------------------------------------------
# 3. Initialize Hugging Face model
# ------------------------------------------------------------

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    max_new_tokens=300
)


# Convert Hugging Face LLM into Chat Model
chat_model = ChatHuggingFace(
    llm=llm
)


# ------------------------------------------------------------
# 4. Few-Shot Prompt Template
# ------------------------------------------------------------

few_shot_template = """
You are a highly professional customer support specialist
at a premium tech company.

Your goal is to address the customer's concerns while
maintaining a calm and helpful demeanor.

Here are some examples of good responses to angry customers:

Angry Customer:
"Your product is terrible! I've been trying to make it work
for hours and nothing helps!"

Response:
"I sincerely apologize for the frustration you're experiencing.
I understand how valuable your time is, and it's unacceptable
that you've spent hours trying to resolve this. Let's work
together to fix this immediately. Could you please tell me
what specific issues you're encountering? I'll guide you
through the solution step by step."

Angry Customer:
"I want a refund right now! This is the worst service ever!"

Response:
"I completely understand your disappointment, and I want to
make this right for you. I'll help you with the refund process
right away. While I process this, could you share what specific
aspects of our service didn't meet your expectations? This will
help us improve and prevent similar issues in the future."

Now, please respond to this customer message:

{customer_message}
"""


few_shot_prompt = PromptTemplate(
    input_variables=["customer_message"],
    template=few_shot_template
)


# ------------------------------------------------------------
# 5. Create LangChain
# ------------------------------------------------------------

few_shot_chain = few_shot_prompt | chat_model


# ------------------------------------------------------------
# 6. Customer Message
# ------------------------------------------------------------

customer_message = """
I've been waiting for 2 hours for your delivery and it still
hasn't arrived!

This is absolutely unacceptable! I paid extra for express
delivery and this is what I get?

I want my money back immediately!
"""


# ------------------------------------------------------------
# 7. Run the Few-Shot Chain
# ------------------------------------------------------------

response = few_shot_chain.invoke(
    {
        "customer_message": customer_message
    }
)


# ------------------------------------------------------------
# 8. Print the response
# ------------------------------------------------------------

print("\n========== AI RESPONSE ==========\n")

print(response.content)