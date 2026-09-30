# ============================================================
# LLM MATH CHAIN USING HUGGING FACE API
# ============================================================

# Install:
# python -m pip install -U langchain langchain-classic
# python -m pip install -U langchain-huggingface
# python -m pip install -U python-dotenv huggingface_hub
# python -m pip install -U numexpr


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import LLMMathChain
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import os


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# Read Hugging Face API key
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_api_key:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found in .env file."
    )


# ============================================================
# 2. INITIALIZE HUGGING FACE MODEL
# ============================================================

llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b:groq",
    huggingfacehub_api_token=hf_api_key,
    task="text-generation",
    max_new_tokens=300,
    temperature=0
)

llm = ChatHuggingFace(
    llm=llm_endpoint
)


# ============================================================
# 3. DIRECT LLM CALCULATION
# ============================================================

result = llm.invoke(
    [
        HumanMessage(
            content="What is 17 raised to the power of 11?"
        )
    ]
)

print("\n\033[1mWhat is 17 raised to the power of 11?\033[0m")
print("-" * 60)
print(result.content)


# ============================================================
# 4. ASK LLM FOR THE PYTHON FORMULA
# ============================================================

result = llm.invoke(
    [
        HumanMessage(
            content=(
                "Give me the Python formula that represents: "
                "What is 17 raised to the power of 11? "
                "Only reply with the formula, nothing else!"
            )
        )
    ]
)

print(
    "\n\033[1mGive me the Python formula that represents: "
    "What is 17 raised to the power of 11?\033[0m"
)

print("-" * 60)
print(result.content)


# ============================================================
# 5. CREATE LLM MATH CHAIN
# ============================================================

llm_math_model = LLMMathChain.from_llm(
    llm=llm,
    verbose=True
)


# ============================================================
# 6. USE LLM MATH CHAIN FOR CALCULATION
# ============================================================

print(
    "\n\033[1mWhat is 17 raised to the power of 11?\033[0m"
)

print("-" * 60)

result = llm_math_model.invoke(
    "What is 17 raised to the power of 11?"
)

print(result)
