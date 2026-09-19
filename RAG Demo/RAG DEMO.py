# pip install langchain
# pip install langchain-text-splitter
# pip install langchain-community 
# pip install langchain-huggingface 
# pip install sentence-transformers 
# pip install chromadb

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Check for the Hugging Face API token
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not hf_token:
    print("WARNING: HUGGINGFACEHUB_API_TOKEN not found in .env file!")

# The newer libraries specifically look for 'HF_TOKEN'
os.environ["HF_TOKEN"] = hf_token

# Step 1: Sample document
print("\nStep 1: Preparing our document")
print("-" * 50)

document = """
Artificial Intelligence (AI) is transforming the way we live and work. Machine learning, 
a subset of AI, enables computers to learn from data without explicit programming. 
Deep learning, a type of machine learning, uses neural networks inspired by the human brain.

Natural Language Processing (NLP) is a branch of AI that helps computers understand and 
process human language. It's used in applications like translation, chatbots, and text analysis.

Computer Vision is another important field in AI. It enables machines to understand and 
process visual information from the world, like images and videos. Applications include 
facial recognition, autonomous vehicles, and medical image analysis.

Reinforcement Learning is a type of machine learning where agents learn by interacting 
with an environment. They receive rewards for good actions and penalties for bad ones. 
This is used in game playing, robotics, and autonomous systems.
"""

print("Document loaded. Length:", len(document), "characters")
print("\nPreview of the document:")
print(document[:200], "...\n")


# Step 2: Text Chunking
print("\nStep 2: Chunking the document")
print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = text_splitter.split_text(document)

print(f"Document has been split into {len(chunks)} chunks.")


# Step 3: Initialize Embeddings
print("\nStep 3: Creating embeddings")
print("-" * 50)

# We use the new langchain_huggingface import here for the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded:", embeddings.model_name)


# Step 4: Create and populate vector store
print("\nStep 4: Creating vector store")
print("-" * 50)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector store created successfully.")


# Step 5: Similarity Search Example
print("\nStep 5: Testing similarity search")
print("-" * 50)

query = "What is reinforcement learning?"
results = vectorstore.similarity_search(query, k=2)

print(f"Query: {query}")
print("\nTop 2 most relevant chunks:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)


# Step 6: Setting up Modern RAG pipeline
print("\nStep 6: Setting up Modern RAG pipeline")
print("-" * 50)

print("Connecting to Hugging Face API...")

# 1. Initialize the LLM (No Chat wrapper needed!)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    do_sample=True,
    huggingfacehub_api_token=hf_token
)

# 2. Create a Standard Prompt Template
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Context: {context}

Question: {input}

Answer:"""

prompt = PromptTemplate.from_template(template)

# 3. Create the Document Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 4. Create the Retrieval Chain
qa_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

# Example questions to ask
questions = [
    "What is reinforcement learning and how does it work?",
    "What are the main applications of computer vision?",
    "How is NLP used in real-world applications?"
]

print("\nAsking questions to our Modern Hugging Face RAG system:")

for question in questions:
    print("\nQuestion:", question)
    try:
        response = qa_chain.invoke({"input": question})
        print("Answer:", response["answer"])
    except Exception as e:
        print("Error getting answer:", str(e))