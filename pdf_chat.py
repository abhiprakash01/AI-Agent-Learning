from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Load PDF
loader = PyPDFLoader("resume.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# Load local LLM
llm = ChatOllama(
    model="qwen3:4b"
)

print("Resume Chat Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    docs = vector_db.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a Resume Assistant.

Answer ONLY from the resume information provided below.

If the answer is not present in the resume, reply:

I could not find that information in the resume.

Resume Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print(f"\nAI: {response.content}\n")