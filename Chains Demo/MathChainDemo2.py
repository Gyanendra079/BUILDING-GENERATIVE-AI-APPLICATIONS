# pip install huggingface_hub
# pip install python-dotenv
# pip install numexpr


from huggingface_hub import InferenceClient
import numexpr as ne

from dotenv import load_dotenv
import os


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# Read the Hugging Face API key
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError(
        "API_KEY not found in .env file."
    )


# ============================================================
# 2. INITIALIZE HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=api_key
)

model = "Qwen/Qwen2.5-72B-Instruct"


# ============================================================
# 3. DEFINE MATH FUNCTION
# ============================================================

def math_chain(question):

    math_prompt = f"""
You are a mathematical assistant that helps solve math problems.

Given a math problem, respond with ONLY a Python expression
that can be evaluated to solve it.

You must start your response with 'Answer: '
followed by the expression.

Do not include any other text or explanations.

For example:

Question: What is 2 plus 2?
Answer: 2 + 2

Question: If I have 3 apples and multiply them by 4,
how many do I have?
Answer: 3 * 4

Question: What is 25 times 4?
Answer: 25 * 4

Question: {question}
"""


    # ========================================================
    # 4. SEND PROMPT TO HUGGING FACE
    # ========================================================

    response = client.chat_completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": math_prompt
            }
        ],
        max_tokens=50,
        temperature=0.01
    )


    # Get model response
    llm_response = response.choices[0].message.content.strip()


    # ========================================================
    # 5. EXTRACT PYTHON EXPRESSION
    # ========================================================

    if "Answer:" in llm_response:

        expression = llm_response.split(
            "Answer:", 1
        )[1].strip()

    else:

        expression = llm_response.strip()


    # Remove Markdown code blocks if generated
    expression = expression.replace(
        "```python", ""
    )

    expression = expression.replace(
        "```", ""
    )

    expression = expression.strip()


    # ========================================================
    # 6. EVALUATE EXPRESSION
    # ========================================================

    try:

        result = ne.evaluate(expression)

    except Exception as e:

        raise ValueError(
            f"Invalid mathematical expression generated "
            f"by the model: {expression}"
        ) from e


    # Return in the same structure as LLMMathChain
    return {
        "answer": str(result)
    }


# ============================================================
# 7. MATH QUESTION
# ============================================================

math_question = "What is 100 times 4?"


# ============================================================
# 8. INVOKE MATH CHAIN
# ============================================================

math_result = math_chain(
    math_question
)


# ============================================================
# 9. DISPLAY OUTPUT
# ============================================================

print(f"Question: {math_question}")
print(f"Answer: {math_result['answer']}")