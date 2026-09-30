import os
import re

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# =========================================================
# 1. Load API key
# =========================================================

load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError(
        "Hugging Face API key not found.\n"
        "Add API_KEY=hf_your_token_here to your .env file."
    )

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key


# =========================================================
# 2. Initialize Hugging Face model
# =========================================================

MODEL_ID = "openai/gpt-oss-120b"

llm_endpoint = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    provider="auto",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
    huggingfacehub_api_token=api_key,
)

llm = ChatHuggingFace(
    llm=llm_endpoint
)


# =========================================================
# 3. Plain-text prompts
# =========================================================

science_prompt = PromptTemplate.from_template(
    """You are a scientific expert.

Answer the question clearly and accurately.

OUTPUT FORMAT RULES:
Return plain text only.
Do not use Markdown.
Do not use bold or italic formatting.
Do not use tables.
Do not use pipe characters.
Do not use LaTeX.
Do not use backslash characters.
Do not use Markdown headings.
Use simple numbered lists if necessary.

Question: {input}

Answer:"""
)


history_prompt = PromptTemplate.from_template(
    """You are a historical expert.

Answer the question clearly and accurately.

OUTPUT FORMAT RULES:
Return plain text only.
Do not use Markdown.
Do not use bold or italic formatting.
Do not use tables.
Do not use pipe characters.
Do not use LaTeX.
Do not use backslash characters.
Do not use Markdown headings.
Use simple numbered lists if necessary.

Question: {input}

Answer:"""
)


math_prompt = PromptTemplate.from_template(
    """You are a mathematics expert.

Solve the problem step by step and explain the calculation clearly.

OUTPUT FORMAT RULES:
Return plain text only.
Do not use Markdown.
Do not use bold or italic formatting.
Do not use tables.
Do not use pipe characters.
Do not use LaTeX.
Do not use backslash characters.
Do not use Markdown headings.
Do not use mathematical LaTeX notation.
Use normal text such as:
sqrt(100) = 10
x^2 = 100

Question: {input}

Answer:"""
)


# =========================================================
# 4. Create destination chains
# =========================================================

science_chain = (
    science_prompt
    | llm
    | StrOutputParser()
)

history_chain = (
    history_prompt
    | llm
    | StrOutputParser()
)

math_chain = (
    math_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# 5. Router prompt
# =========================================================

router_prompt = PromptTemplate.from_template(
    """Classify the question into exactly ONE category.

Categories:
Science
History
Math

Rules:
Science = biology, chemistry, physics, environment,
astronomy, medicine and scientific concepts.

History = historical people, events, dates,
civilizations, wars and historical facts.

Math = arithmetic, algebra, geometry, calculus,
statistics, equations and mathematical calculations.

Return ONLY one of these three words:
Science
History
Math

Do not provide an explanation.

Question: {input}

Category:"""
)


router_chain = (
    router_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# 6. Clean model output
# =========================================================

def clean_output(text):

    # Remove Markdown bold
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Remove Markdown italic
    text = re.sub(r"(?<!\*)\*(.*?)\*(?!\*)", r"\1", text)

    # Remove Markdown headings
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Remove LaTeX display markers
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")

    # Remove LaTeX commands commonly generated
    text = text.replace("\\boxed", "")
    text = text.replace("\\text", "")

    # Remove pipe characters used in tables
    text = text.replace("|", " ")

    # Remove backslashes
    text = text.replace("\\", "")

    # Clean excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Clean excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


# =========================================================
# 7. Routing function
# =========================================================

def route_chain(input_text):

    category = router_chain.invoke(
        {
            "input": input_text
        }
    ).strip().lower()

    # Remove accidental Markdown/punctuation
    category = category.replace("*", "")
    category = category.replace("#", "")
    category = category.replace(":", "")
    category = category.strip()

    print(f"Router output: {category}")

    if "science" in category:

        print("[Routed to Science]")
        return science_chain

    elif "history" in category:

        print("[Routed to History]")
        return history_chain

    elif "math" in category:

        print("[Routed to Math]")
        return math_chain

    else:

        raise ValueError(
            f"Router returned an invalid category: {category}"
        )


# =========================================================
# 8. Test questions
# =========================================================

questions = [
    "What causes photosynthesis?",
    "Who was the first president of the United States?",
    "What is the square root of 100?"
]


# =========================================================
# 9. Run router
# =========================================================

for question in questions:

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)

    try:

        chain = route_chain(question)

        answer = chain.invoke(
            {
                "input": question
            }
        )

        answer = clean_output(answer)

        print("\nAnswer:")
        print(answer)

    except Exception as e:

        print("\nError:")
        print(e)