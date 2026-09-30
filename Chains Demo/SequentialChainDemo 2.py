# ============================================================
# SEQUENTIAL CHAIN USING HUGGING FACE INFERENCE PROVIDERS
# ============================================================

# Install:
# python -m pip install -U langchain langchain-classic
# python -m pip install -U langchain-huggingface
# python -m pip install -U python-dotenv huggingface_hub


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from dotenv import load_dotenv
import os


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

hf_api_key = os.getenv("API_KEY")

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
    temperature=0.7
)

llm = ChatHuggingFace(
    llm=llm_endpoint
)


# ============================================================
# 3. FIRST CHAIN → GENERATE ACADEMIC TOPIC
# ============================================================

topic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant who generates interesting "
        "academic topics."
    ),
    (
        "user",
        "Generate one random academic topic for discussion. "
        "Return only the topic title."
    )
])

topic_chain = (
    topic_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 4. SECOND CHAIN → GENERATE QUESTIONS
# ============================================================

question_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant who generates "
        "thought-provoking academic questions."
    ),
    (
        "user",
        "Generate exactly 3 thought-provoking questions "
        "about this topic:\n\n{topic}"
    )
])

question_chain = (
    question_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 5. CREATE SEQUENTIAL CHAIN
# ============================================================

sequential_chain = (
    {"topic": topic_chain}
    | RunnablePassthrough()
    | {
        "topic": itemgetter("topic"),
        "questions": question_chain
    }
)


# ============================================================
# 6. RUN THE SEQUENTIAL CHAIN
# ============================================================

results = sequential_chain.invoke({})


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GENERATE 3 QUESTIONS ABOUT A RANDOM ACADEMIC TOPIC")
print("=" * 60)

print("\nAcademic Topic:")
print(results["topic"])

print("\nThought-Provoking Questions:")
print(results["questions"])