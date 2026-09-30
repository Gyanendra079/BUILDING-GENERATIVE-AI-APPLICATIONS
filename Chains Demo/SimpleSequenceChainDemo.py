# ============================================================
# SIMPLE SEQUENTIAL CHAIN USING HUGGING FACE
# ============================================================

# Install:
# python -m pip install -U langchain-huggingface langchain-core python-dotenv


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. INITIALIZE HUGGING FACE MODEL
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=700,
    temperature=0.7
)

chat_model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# 3. CHAIN ONE - GENERATE BLOG OUTLINE
# ============================================================

template_one = """
You are a professional content writer.

Create a simple and well-organized outline for a beginner-friendly
blog post about:

{topic}

FORMAT REQUIREMENTS:

1. Use numbered headings.
2. Put every heading on a separate line.
3. Use short bullet points under each heading.
4. Keep the outline concise.
5. Do not include Python code.
6. Do not include LangChain objects.
7. Do not include metadata.
8. Return ONLY the outline.
"""

first_prompt = ChatPromptTemplate.from_template(
    template_one
)


# ============================================================
# 4. CREATE CHAIN ONE
# ============================================================

chain_one = (
    first_prompt
    | chat_model
    | StrOutputParser()
)


# ============================================================
# 5. INPUT TOPIC
# ============================================================

# topic = "Artificial Intelligence"
topic = "Travel"


# ============================================================
# 6. RUN CHAIN ONE
# ============================================================

chain_one_result = chain_one.invoke(
    {
        "topic": topic
    }
)

chain_one_result = str(chain_one_result).strip()


# ============================================================
# 7. DISPLAY OUTLINE
# ============================================================

print("\n")
print("\033[1mGive me a simple bullet point outline for a blog post\033[0m")
print("-" * 60)
print(chain_one_result)


# ============================================================
# 8. CHAIN TWO - WRITE BLOG POST
# ============================================================

template_two = """
You are a professional blog writer.

Write a complete beginner-friendly blog post using the outline
provided below.

OUTLINE:

{abc}

============================================================

FORMATTING REQUIREMENTS:

1. Start with a clear title.

2. Format the title as a Markdown H1 heading.

3. Use numbered Markdown H2 headings for the main sections.

4. Example section format:

## 1. Introduction

## 2. What Is Artificial Intelligence?

## 3. Types of Artificial Intelligence

5. Each section must contain properly separated paragraphs.

6. Use bullet points where they improve readability.

7. Use short paragraphs.

8. Use correct spelling and grammar.

9. Explain technical concepts in beginner-friendly language.

10. Do not repeat the outline separately before the article.

11. Do not mention that you are an AI.

12. Do not output Python objects.

13. Do not output LangChain objects.

14. Do not output metadata.

15. Do not output debugging information.

16. Do not output text such as:
    types
    partial_variables
    ChatPromptTemplate
    ChatOpenAI
    HuggingFaceEndpoint

17. Do not include corrupted or random text.

18. Finish with a clear conclusion.

19. Return ONLY the final blog post.

============================================================

Write the blog post now.
"""


second_prompt = ChatPromptTemplate.from_template(
    template_two
)


# ============================================================
# 9. CREATE FULL SEQUENTIAL CHAIN
# ============================================================

full_chain = (
    {"abc": chain_one}
    | second_prompt
    | chat_model
    | StrOutputParser()
)


# ============================================================
# 10. RUN COMPLETE CHAIN
# ============================================================

result = full_chain.invoke(
    {
        "topic": topic
    }
)

result = str(result).strip()


# ============================================================
# 11. DISPLAY FINAL BLOG POST
# ============================================================

print("\n")
print("\033[1mWrite a blog post using the outline\033[0m")
print("-" * 60)
print(result)
print("\n")
