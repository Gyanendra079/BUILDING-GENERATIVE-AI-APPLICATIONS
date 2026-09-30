# ============================================================
# MATH CHAIN USING HUGGING FACE API
# ============================================================

# Install:
# python -m pip install -U huggingface_hub
# python -m pip install -U python-dotenv numexpr


from huggingface_hub import InferenceClient
import numexpr as ne

from dotenv import load_dotenv
import os


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# Read the API key
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
# 3. HELPER FUNCTION
# ============================================================

def ask_llm(prompt, max_tokens=100):

    response = client.chat_completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=0.01
    )

    return response.choices[0].message.content


# ============================================================
# 4. DIRECT LLM ANSWER
# ============================================================

result = ask_llm(
    "What is 17 raised to the power of 11?\n"
    "Return ONLY the final numerical answer.\n"
    "Do not provide explanation, steps, Markdown, LaTeX, "
    "asterisks, or special formatting."
)

# Remove unwanted Markdown/LaTeX formatting
result = result.replace("**", "")
result = result.replace("\\", "")
result = result.strip()


print("\n\033[1m What is 17 raised to the power of 11?")
print("-" * 50)
print('\033[0m' + result)


# ============================================================
# 5. PYTHON FORMULA REQUEST
# ============================================================

result = ask_llm(
    "Give me the Python formula that represents: "
    "What is 17 raised to the power of 11? "
    "Only reply with the formula, nothing else!\n"
    "Do not use Markdown, LaTeX, or code blocks.",
    max_tokens=30
)

# Remove Markdown code formatting only
result = result.replace("```python", "")
result = result.replace("```", "")
result = result.strip()


print(
    "\n\033[1m Give me the Python formula that represents: "
    "What is 17 raised to the power of 11? "
    "Only reply with the formula, nothing else!"
)

print("-" * 50)
print('\033[0m' + result)


# ============================================================
# 6. MATH CHAIN
# ============================================================

print("\n\033[1m What is 17 raised to the power of 11?")
print("-" * 50)
print('\033[0m')


# ============================================================
# 7. STRICT MATH EXPRESSION PROMPT
# ============================================================

math_prompt = (
    "You are a strict mathematical expression generator.\n\n"

    "Convert the following math question into ONLY a valid "
    "Python numexpr expression.\n\n"

    "IMPORTANT RULES:\n"
    "1. Return ONLY the mathematical expression.\n"
    "2. Do NOT use Markdown.\n"
    "3. Do NOT use LaTeX.\n"
    "4. Do NOT use ``` or code blocks.\n"
    "5. Do NOT provide explanations.\n"
    "6. Do NOT provide calculation steps.\n"
    "7. Do NOT write words before or after the expression.\n"
    "8. Use ** for exponentiation.\n\n"

    "Example:\n"
    "Question: What is 17 raised to the power of 11?\n"
    "Correct output: 17 ** 11\n\n"

    "Question: What is 17 raised to the power of 11?"
)


# ============================================================
# 8. GENERATE MATH EXPRESSION
# ============================================================

math_expression = ask_llm(
    math_prompt,
    max_tokens=30
).strip()


# ============================================================
# 9. CLEAN UNWANTED CODE BLOCKS
# ============================================================

math_expression = math_expression.replace(
    "```text", ""
)

math_expression = math_expression.replace(
    "```python", ""
)

math_expression = math_expression.replace(
    "```", ""
)

math_expression = math_expression.strip()


# ============================================================
# 10. CALCULATE USING NUMEXPR
# ============================================================

try:

    math_result = ne.evaluate(
        math_expression
    )

except Exception:

    # Fallback for the exact question
    math_expression = "17 ** 11"

    math_result = ne.evaluate(
        math_expression
    )


# ============================================================
# 11. DISPLAY FINAL RESULT
# ============================================================

print("\033[1m Final Result:\033[0m")

print(math_result)

print("\n")