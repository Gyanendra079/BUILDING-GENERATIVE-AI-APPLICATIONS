# Install:
# python -m pip install -U huggingface_hub python-dotenv
#
# ============================================================


import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# 1. LOAD HUGGING FACE API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError(
        "API_KEY was not found.\n"
        "Please add your Hugging Face token to the .env file."
    )


# ============================================================
# 2. CREATE HUGGING FACE INFERENCE CLIENT
# ============================================================

client = InferenceClient(
    api_key=api_key
)


# ============================================================
# 3. MODEL
# ============================================================

MODEL = "openai/gpt-oss-120b:cheapest"


# ============================================================
# 4. COMMON FUNCTION FOR ALL THREE CHAINS
# ============================================================

def generate_response(prompt):
    """
    Send a prompt to Hugging Face and return
    the generated text.
    """

    response = client.chat_completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 5. EMPLOYEE REVIEW INPUT
# ============================================================

employee_review = '''
Employee Information:
Name: Joe Smith
Position: Software Engineer
Date of Review: Jan 01, 2000

Strengths:
Joe is a highly skilled software engineer with a deep understanding of programming languages, algorithms, and software development best practices. His technical expertise shines through in his ability to efficiently solve complex problems and deliver high-quality code.

One of Joe's greatest strengths is his collaborative nature. He actively engages with cross-functional teams, contributing valuable insights and seeking input from others. His open-mindedness and willingness to learn from colleagues make him a true team player.

Joe consistently demonstrates initiative and self-motivation. He takes the lead in seeking out new projects and challenges, and his proactive attitude has led to significant improvements in existing processes and systems. His dedication to self-improvement and growth is commendable.

Another notable strength is Joe's adaptability. He has shown great flexibility in handling changing project requirements and learning new technologies. This adaptability allows him to seamlessly transition between different projects and tasks, making him a valuable asset to the team.

Joe's problem-solving skills are exceptional. He approaches issues with a logical mindset and consistently finds effective solutions, often thinking outside the box. His ability to break down complex problems into manageable parts is key to his success in resolving issues efficiently.

Weaknesses:
While Joe possesses numerous strengths, there are a few areas where he could benefit from improvement. One such area is time management. Occasionally, Joe struggles with effectively managing his time, resulting in missed deadlines or the need for additional support to complete tasks on time. Developing better prioritization and time management techniques would greatly enhance his efficiency.

Another area for improvement is Joe's written communication skills. While he communicates well verbally, there have been instances where his written documentation lacked clarity, leading to confusion among team members. Focusing on enhancing his written communication abilities will help him effectively convey ideas and instructions.

Additionally, Joe tends to take on too many responsibilities and hesitates to delegate tasks to others. This can result in an excessive workload and potential burnout. Encouraging him to delegate tasks appropriately will not only alleviate his own workload but also foster a more balanced and productive team environment.
'''


# ============================================================
# CHAIN 1
# EMPLOYEE REVIEW → SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHAIN 1 → EMPLOYEE PERFORMANCE SUMMARY")
print("=" * 70)

prompt1 = f"""
Summarize the following employee performance review.

IMPORTANT OUTPUT RULES:
- Return plain text only.
- Do NOT use Markdown.
- Do NOT use ** symbols.
- Do NOT use tables.
- Do NOT use pipes.
- Do NOT use emojis.
- Write one concise paragraph.
- Mention both strengths and areas for improvement.

Employee Performance Review:

{employee_review}
"""

summary = generate_response(prompt1)

print("\nINPUT:")
print("Employee Performance Review")

print("\nOUTPUT:")
print(summary)


# ============================================================
# CHAIN 2
# SUMMARY → WEAKNESSES
# ============================================================

print("\n" + "=" * 70)
print("CHAIN 2 → IDENTIFY EMPLOYEE WEAKNESSES")
print("=" * 70)

prompt2 = f"""
Identify the key employee weaknesses from the following
performance review summary.

IMPORTANT OUTPUT RULES:
- Return plain text only.
- Do NOT use Markdown.
- Do NOT use tables.
- Do NOT use pipes.
- Do NOT use emojis.
- Return exactly 3 numbered points.
- Mention the weakness and briefly explain it.

Performance Review Summary:

{summary}
"""

weaknesses = generate_response(prompt2)

print("\nINPUT:")
print("Output of Chain 1: Summary")

print("\nOUTPUT:")
print(weaknesses)


# ============================================================
# CHAIN 3
# WEAKNESSES → PERSONALIZED ACTION PLAN
# ============================================================

print("\n" + "=" * 70)
print("CHAIN 3 → PERSONALIZED ACTION PLAN")
print("=" * 70)

prompt3 = f"""
Create a personalized action plan to address the following
employee weaknesses.

IMPORTANT OUTPUT RULES:
- Return plain text only.
- Do NOT use Markdown.
- Do NOT use tables.
- Do NOT use pipes.
- Do NOT use emojis.
- Use exactly 3 numbered sections.
- For each weakness provide:
  Weakness:
  Action:
  Expected Improvement:

Employee Weaknesses:

{weaknesses}
"""

plan = generate_response(prompt3)

print("\nINPUT:")
print("Output of Chain 2: Employee Weaknesses")

print("\nOUTPUT:")
print(plan)


# ============================================================
# FINAL SEQUENTIAL CHAIN RESULT
# ============================================================

print("\n" + "=" * 70)
print("SEQUENTIAL CHAIN COMPLETED")
print("=" * 70)

print("""
Chain 1: Employee Review → Summary
        ↓
Chain 2: Summary → Weaknesses
        ↓
Chain 3: Weaknesses → Action Plan
        ↓
Final Result
""")

print("FINAL ACTION PLAN:")
print(plan)