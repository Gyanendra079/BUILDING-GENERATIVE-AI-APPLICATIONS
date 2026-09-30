# ============================================================
# CrewAI Use Case 1
# Web Scraping + Question Answering
#
# CrewAI + Hugging Face
# Custom BaseLLM implementation
#
# No OPENAI_API_KEY required
# ============================================================

# Install if required:
# python -m pip install -U crewai crewai-tools python-dotenv openai


import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from crewai import Agent, Task, Crew, BaseLLM
from crewai_tools import ScrapeWebsiteTool


# ============================================================
# 1. PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

TEXT_FILE = BASE_DIR / "ai.txt"


# ============================================================
# 2. LOAD .ENV
# ============================================================

load_dotenv(dotenv_path=ENV_FILE)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError(
        f"API_KEY was not found.\n\n"
        f"Expected .env file at:\n{ENV_FILE}\n\n"
        f"Your .env should contain:\n"
        f"API_KEY=your_huggingface_token"
    )

print("Hugging Face API key loaded successfully.")


# ============================================================
# 3. CUSTOM HUGGING FACE LLM
# ============================================================

class HuggingFaceLLM(BaseLLM):
    """
    Custom CrewAI LLM that directly calls the
    Hugging Face OpenAI-compatible API.

    This bypasses CrewAI's built-in OpenAI model
    name handling.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2
    ):

        super().__init__(
            model=model,
            temperature=temperature
        )

        self.api_key = api_key

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )

    def call(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return response.choices[0].message.content


# ============================================================
# 4. CREATE HUGGING FACE LLM
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING HUGGING FACE LLM")
print("=" * 70)

llm = HuggingFaceLLM(
    api_key=API_KEY,

    # IMPORTANT:
    # Keep the complete Hugging Face model ID.
    model="openai/gpt-oss-120b:cerebras",

    temperature=0.2
)

print("Hugging Face LLM configured successfully.")
print("Model: openai/gpt-oss-120b:cerebras")


# ============================================================
# 5. SCRAPE WIKIPEDIA
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: SCRAPING WIKIPEDIA")
print("=" * 70)

scrape_tool = ScrapeWebsiteTool(
    website_url="https://en.wikipedia.org/wiki/Artificial_intelligence"
)

text = scrape_tool.run()

print("Wikipedia data scraped successfully.")


# ============================================================
# 6. SAVE SCRAPED DATA
# ============================================================

with open(TEXT_FILE, "w", encoding="utf-8") as file:
    file.write(text)

print("\nScraped data saved to:")
print(TEXT_FILE)


# ============================================================
# 7. READ SCRAPED DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: READING SCRAPED DATA")
print("=" * 70)

with open(TEXT_FILE, "r", encoding="utf-8") as file:
    context = file.read()

print(f"Total characters loaded: {len(context):,}")


# ============================================================
# 8. FIND RELEVANT NLP CONTEXT
# ============================================================

question = "What is Natural Language Processing?"

keywords = [
    "natural language processing",
    "natural-language processing",
    "NLP"
]

paragraphs = context.split("\n")

relevant_sections = []

for paragraph in paragraphs:

    paragraph_lower = paragraph.lower()

    if any(
        keyword.lower() in paragraph_lower
        for keyword in keywords
    ):
        paragraph = paragraph.strip()

        if paragraph:
            relevant_sections.append(paragraph)


# ============================================================
# 9. CREATE RETRIEVED CONTEXT
# ============================================================

retrieved_context = "\n\n".join(relevant_sections)


# ============================================================
# 10. FALLBACK
# ============================================================

if not retrieved_context:

    print(
        "Specific NLP information was not found."
    )

    print(
        "Using the beginning of the scraped document."
    )

    retrieved_context = context[:15000]


# Keep context manageable
retrieved_context = retrieved_context[:15000]

print(
    f"Relevant context characters: "
    f"{len(retrieved_context):,}"
)


# ============================================================
# 11. CREATE CREWAI AGENT
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: CREATING CREWAI AGENT")
print("=" * 70)

data_analyst = Agent(

    role="AI Educator",

    goal=(
        "Explain Natural Language Processing accurately "
        "using the provided context."
    ),

    backstory=(
        "You are an experienced AI and Data Science educator "
        "who explains technical concepts clearly and simply."
    ),

    llm=llm,

    verbose=True,

    allow_delegation=False
)


# ============================================================
# 12. CREATE TASK
# ============================================================

test_task = Task(

    description=f"""
    Answer the following question:

    {question}

    Use the following retrieved context:

    ------------------------------
    RETRIEVED CONTEXT
    ------------------------------

    {retrieved_context}

    ------------------------------
    END OF CONTEXT
    ------------------------------

    Explain:

    1. What Natural Language Processing is.
    2. What NLP allows computers to do.
    3. Give a few common applications of NLP.

    Provide a clear and concise answer.

    Do not invent information that is not supported
    by the provided context.
    """,

    agent=data_analyst,

    expected_output=(
        "A clear explanation of Natural Language Processing, "
        "including its purpose and common applications."
    )
)


# ============================================================
# 13. CREATE CREW
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: CREATING CREW")
print("=" * 70)

crew = Crew(

    agents=[data_analyst],

    tasks=[test_task],

    verbose=True
)


# ============================================================
# 14. RUN CREW
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: RUNNING CREWAI")
print("=" * 70)

result = crew.kickoff()


# ============================================================
# 15. DISPLAY FINAL ANSWER
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)

print(result.raw)

print("=" * 70)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)