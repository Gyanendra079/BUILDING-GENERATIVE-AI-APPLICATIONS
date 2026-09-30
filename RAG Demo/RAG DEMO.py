# ============================================================
# SIMPLE RAG DEMO USING HUGGING FACE + CHROMA
# ============================================================
#
# Install:
#
# python -m pip install -U langchain
# python -m pip install -U langchain-text-splitters
# python -m pip install -U langchain-community
# python -m pip install -U langchain-huggingface
# python -m pip install -U sentence-transformers
# python -m pip install -U chromadb
# python -m pip install -U huggingface-hub
# python -m pip install -U python-dotenv
#
# ============================================================


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from huggingface_hub import InferenceClient

from dotenv import load_dotenv
import os


# ============================================================
# STEP 1: LOAD HUGGING FACE API KEY
# ============================================================

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN not found in .env file"
    )

print("Hugging Face API key loaded successfully.")


# ============================================================
# STEP 2: SAMPLE DOCUMENT
# ============================================================

document = """
Artificial Intelligence (AI) is a branch of computer science
that focuses on creating systems capable of performing tasks
that normally require human intelligence.

Machine Learning (ML) is a subset of Artificial Intelligence.
It enables computers to learn patterns from data and make
predictions or decisions without being explicitly programmed
for every task.

Deep Learning is a subset of Machine Learning that uses
artificial neural networks with multiple layers to learn
complex patterns from large amounts of data.

Natural Language Processing (NLP) is a field of Artificial
Intelligence that focuses on enabling computers to understand,
interpret, and generate human language.

NLP is used in applications such as chatbots, language
translation, sentiment analysis, text summarization,
spam detection, and voice assistants.

Computer Vision is another field of Artificial Intelligence
that enables computers to understand and analyze images
and videos.

The main applications of computer vision include facial
recognition, medical image analysis, autonomous vehicles,
object detection, surveillance, quality inspection, and
image classification.

Reinforcement Learning is a type of Machine Learning in which
an agent learns by interacting with an environment. The agent
receives rewards for desirable actions and penalties for
undesirable actions.

Reinforcement Learning is commonly used in robotics, game
playing, autonomous systems, recommendation systems, and
decision-making problems.
"""


print("\nDocument loaded. Length:", len(document), "characters")

print("\nPreview of the document:")
print(document[:200], "...\n")


# ============================================================
# STEP 3: TEXT CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = text_splitter.split_text(document)

print(
    f"Document has been split into {len(chunks)} chunks."
)

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i + 1}:")
    print(chunk)
    print("-" * 30)


# ============================================================
# STEP 4: EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(
    "\nEmbedding model loaded:",
    "sentence-transformers/all-MiniLM-L6-v2"
)

print(
    "This model will convert text chunks into numerical vectors"
)


# ============================================================
# STEP 5: VECTOR STORE
# ============================================================

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\nVector store created with following details:")

print(f"- Number of texts: {len(chunks)}")

print(
    f"- Embedding dimension: "
    f"{len(embeddings.embed_query('test'))}"
)

print("- Database location: ./chroma_db")


# ============================================================
# STEP 6: SIMILARITY SEARCH
# ============================================================

query = "What is reinforcement learning?"

results = vectorstore.similarity_search(
    query,
    k=2
)

print(f"\nQuery: {query}")

print("\nTop 2 most relevant chunks:")

for i, doc in enumerate(results):

    print(f"\nResult {i + 1}:")
    print(doc.page_content)

    print("-" * 30)


# ============================================================
# STEP 7: HUGGING FACE INFERENCE CLIENT
# ============================================================

client = InferenceClient(
    api_key=hf_token
)


# ============================================================
# STEP 8: SELECT MODEL
# ============================================================

model = "Qwen/Qwen3-8B"

print("\nHugging Face model:", model)


# ============================================================
# STEP 9: RAG FUNCTION
# ============================================================

def ask_rag(question):

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    relevant_docs = vectorstore.similarity_search(
        question,
        k=2
    )

    # --------------------------------------------------------
    # Combine retrieved chunks
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )

    # --------------------------------------------------------
    # Create RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I don't know based on the provided document."

Do not invent information.

Context:
{context}

Question:
{question}
"""

    # --------------------------------------------------------
    # Hugging Face Chat Completion
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=256,

        temperature=0.2
    )

    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    answer = response.choices[0].message.content

    return answer


# ============================================================
# STEP 10: ASK QUESTIONS TO RAG SYSTEM
# ============================================================

questions = [

    "What is reinforcement learning and how does it work?",

    "What are the main applications of computer vision?",

    "How is NLP used in real-world applications?"

]


print("\nAsking questions to our Hugging Face RAG system:")


for question in questions:

    print("\nQuestion:", question)

    try:

        answer = ask_rag(question)

        print("Answer:", answer)

    except Exception as e:

        print(
            "Error getting answer:",
            str(e)
        )  