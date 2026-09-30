# ============================================================
# RAG CHAIN APPLICATION USING HUGGING FACE
# ============================================================

# Install:
# python -m pip install -U langchain-text-splitters
# python -m pip install -U langchain-huggingface
# python -m pip install -U langchain-community
# python -m pip install -U chromadb
# python -m pip install -U huggingface_hub
# python -m pip install -U python-dotenv


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

import os


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# Make sure your .env contains:
#
# API_KEY=hf_your_token_here

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError(
        "API_KEY not found in .env file."
    )


# ============================================================
# 3. INITIALIZE HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    api_key=api_key
)

# Hugging Face chat model
model = "Qwen/Qwen2.5-72B-Instruct"


# ============================================================
# 4. DEFINE HUGGING FACE LLM FUNCTION
# ============================================================

def generate_response(prompt_text):

    # prompt_text is already a normal Python string.
    # ChatPromptValue was converted to string before
    # reaching this function.

    response = client.chat_completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        max_tokens=256,
        temperature=0.1
    )

    return response.choices[0].message.content


# Convert the function into a LangChain Runnable
llm = RunnableLambda(
    generate_response
)


# ============================================================
# 5. DEFINE INPUT DOCUMENT
# ============================================================

documents = [
    "Apple printer sales are up by 30% in 2011 Q4",
    "Lenovo Laptop sales are down by 10% in 2012",
    "A3 printer sales is 5% more in middle east as compared to A4 total"
]


# ============================================================
# 6. CREATE TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

splits = text_splitter.create_documents(
    documents
)


# ============================================================
# 7. DISPLAY DOCUMENT CHUNKS
# ============================================================

print(
    f"Document has been split into {len(splits)} chunks."
)

print("\n\033[1mExample chunks:")
print("-" * 50)
print("\033[0m")


for i, chunk in enumerate(splits):

    print(f"\nChunk {i + 1}:")
    print(chunk.page_content)
    print("-" * 30)


# ============================================================
# 8. DEFINE HUGGING FACE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=api_key
)


# ============================================================
# 9. CREATE VECTOR STORE
# ============================================================

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)


# ============================================================
# 10. CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever()


# ============================================================
# 11. CREATE RAG TEMPLATE
# ============================================================

template = """Answer the following question based on the provided context. If the context does not contain the answer, say "I don't know based on the context provided."

Context: {context}
Question: {question}

Answer:"""


# ============================================================
# 12. CREATE RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    template
)


# ============================================================
# 13. FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_documents(docs):

    return "\n\n".join(
        document.page_content
        for document in docs
    )


# ============================================================
# 14. CONVERT PROMPT TO STRING
# ============================================================

def prompt_to_string(prompt_value):

    return prompt_value.to_string()


# ============================================================
# 15. CREATE RAG CHAIN
# ============================================================

rag_chain = (
    {
        "context": (
            retriever
            | RunnableLambda(format_documents)
        ),

        "question": RunnablePassthrough()
    }

    # Create ChatPromptValue
    | prompt

    # ChatPromptValue → Python string
    | RunnableLambda(prompt_to_string)

    # Python string → Hugging Face
    | llm

    # Final output → string
    | StrOutputParser()
)


# ============================================================
# 16. EXECUTE QUERY
# ============================================================

question = "What is Apple printer sales in 2011 Q4?"

answer = rag_chain.invoke(
    question
)


# ============================================================
# 17. DISPLAY FINAL OUTPUT
# ============================================================

print(
    f"\n\033[1m Question: {question}"
)

print(
    f"\033[0m Answer: {answer.strip()}"
)
