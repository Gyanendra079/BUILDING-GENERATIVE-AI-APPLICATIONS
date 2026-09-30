# ============================================================
# MEMORY DEMO USING LOCAL HUGGING FACE MODEL
# ============================================================

# Install:
#
# python -m pip install -U transformers
# python -m pip install -U torch
# python -m pip install -U accelerate
# python -m pip install -U langchain
# python -m pip install -U langchain-classic
# python -m pip install -U python-dotenv
#
# ============================================================


from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_classic.memory import ConversationBufferMemory

from dotenv import load_dotenv

import os
import torch


# ============================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ============================================================

env_path = os.path.join(
    os.path.dirname(__file__),
    ".env"
)

load_dotenv(env_path)


# ============================================================
# STEP 2: MODEL
# ============================================================

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("\nLoading Hugging Face model:")
print(model_name)

print("\nFirst run may take some time because the model")
print("needs to be downloaded to your computer.")


# ============================================================
# STEP 3: LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# ============================================================
# STEP 4: LOAD MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


print("\nHugging Face model loaded successfully.")


# ============================================================
# STEP 5: MEMORY
# ============================================================

memory = ConversationBufferMemory(
    return_messages=True
)


# ============================================================
# STEP 6: FUNCTION TO GENERATE RESPONSE
# ============================================================

def chat_with_memory(user_input):

    # --------------------------------------------------------
    # Get previous conversation
    # --------------------------------------------------------

    history = memory.load_memory_variables({})["history"]


    # --------------------------------------------------------
    # Build conversation messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Answer the user's current question clearly "
                "and concisely. "
                "Use previous conversation only when needed "
                "to understand the context."
            )
        }
    ]


    # --------------------------------------------------------
    # Add previous messages
    # --------------------------------------------------------

    for message in history:

        if message.type == "human":

            messages.append(
                {
                    "role": "user",
                    "content": message.content
                }
            )

        elif message.type == "ai":

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content
                }
            )


    # --------------------------------------------------------
    # Add current user question
    # --------------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # --------------------------------------------------------
    # Convert messages into model input
    # --------------------------------------------------------

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.05
        )


    # --------------------------------------------------------
    # Remove original prompt from output
    # --------------------------------------------------------

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    # --------------------------------------------------------
    # Decode response
    # --------------------------------------------------------

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------------
    # Save conversation in memory
    # --------------------------------------------------------

    memory.save_context(
        {"input": user_input},
        {"output": response}
    )


    return response


# ============================================================
# STEP 7: START CONVERSATION
# ============================================================

print("\n" + "=" * 60)
print("CONVERSATION BUFFER MEMORY")
print("=" * 60)


print("\nHuman: Hi, my name is Gyanendra")

response = chat_with_memory(
    "Hi, my name is Gyanendra"
)

print("AI:", response)


print(
    "\nHuman: I'm doing well! Just having a conversation "
    "with an AI."
)

response = chat_with_memory(
    "I'm doing well! Just having a conversation with an AI."
)

print("AI:", response)


print("\nHuman: What is AI?")

response = chat_with_memory(
    "What is AI?"
)

print("AI:", response)


print("\nHuman: What is my name?")

response = chat_with_memory(
    "What is my name?"
)

print("AI:", response)


# ============================================================
# STEP 8: DISPLAY MEMORY
# ============================================================

print("\n" + "=" * 60)
print("CONVERSATION MEMORY")
print("=" * 60)

for message in memory.buffer:

    if message.type == "human":

        print(
            f"Human: {message.content}"
        )

    elif message.type == "ai":

        print(
            f"AI: {message.content}"
        )