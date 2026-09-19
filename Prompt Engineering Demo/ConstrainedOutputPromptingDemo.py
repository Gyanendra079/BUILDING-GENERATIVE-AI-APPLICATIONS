# ============================================================
# HUGGING FACE CONSTRAINED GENERATION
# ============================================================

# Install:
# py -3.14 -m pip install -U openai python-dotenv langchain-core


import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. LOAD .ENV
# ============================================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN was not found.\n"
        "Check your .env file."
    )

print("Hugging Face token loaded successfully")


# ============================================================
# 2. HUGGING FACE CLIENT
# ============================================================

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token
)


# ============================================================
# 3. MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b:groq"


# ============================================================
# 4. PROMPT
# ============================================================

constrained_template = """
You are a customer support specialist.

Generate a response to the customer's message following
these strict constraints:

1. Response must be exactly 3 paragraphs long.
2. First paragraph must be exactly 2 sentences showing empathy.
3. Second paragraph must provide exactly 2 concrete solutions.
4. Third paragraph must end with a question to engage the customer.
5. Total response must maintain a professional tone.
6. Must not use any exclamation marks.
7. Must include exactly one apology.
8. Must use the phrase "I understand" exactly once.

Important:
Follow all constraints exactly.

Do not use headings.
Do not use bullet points.
Do not use numbering.
Do not add extra text.

Customer message:

{customer_message}
"""


# ============================================================
# 5. LANGCHAIN PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=["customer_message"],
    template=constrained_template
)


# ============================================================
# 6. CUSTOMER MESSAGE
# ============================================================

customer_message = """
I've been waiting for 2 hours for your delivery and it still
hasn't arrived.

This is absolutely unacceptable.

I paid extra for express delivery and this is what I get?

I want my money back immediately.
"""


# ============================================================
# 7. CREATE FINAL PROMPT
# ============================================================

final_prompt = prompt.format(
    customer_message=customer_message
)


print("\nSending request to Hugging Face...")
print("Model:", MODEL_NAME)


# ============================================================
# 8. CALL HUGGING FACE
# ============================================================

try:

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],

        temperature=0.7,

        max_tokens=1000
    )


    # ========================================================
    # 9. CHECK RESPONSE
    # ========================================================

    print("\nAPI request successful")

    choice = response.choices[0]

    message = choice.message


    # ========================================================
    # 10. GET NORMAL CONTENT
    # ========================================================

    answer = message.content


    # ========================================================
    # 11. DISPLAY RESPONSE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("HUGGING FACE CONSTRAINED CUSTOMER SUPPORT RESPONSE")
    print("=" * 70)

    if answer:

        print(answer)

    else:

        print("The model returned no normal content.")
        print("\nComplete message returned by API:")
        print(message)


    print("=" * 70)


except Exception as e:

    print("\n")
    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print(type(e).__name__)
    print(str(e))

    print("=" * 70)

