from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# -------------------------
# Load Resume PDF
# -------------------------

loader = PyPDFLoader("resume.pdf")
documents = loader.load()

# -------------------------
# Split into Chunks
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# -------------------------
# Create Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Create Vector Database
# -------------------------

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# -------------------------
# Load Local LLM
# -------------------------

llm = ChatOllama(
    model="qwen3:4b"
)

print("\nCareer Assistant Ready!")
print("Type 'exit' to quit.\n")

# -------------------------
# Chat Loop
# -------------------------

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    docs = vector_db.similarity_search(
        user_input,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # -------------------------
    # Resume Analysis
    # -------------------------

    if "analyze" in user_input.lower():

        prompt = f"""
You are an expert career coach.

Analyze the following resume and provide:

1. Top 3 strengths
2. Top 3 areas for improvement
3. Top 3 skills to learn next

Resume:
{context}
"""

    # -------------------------
    # Interview Questions
    # -------------------------

    elif "interview" in user_input.lower():

        prompt = f"""
Based on the following resume,
generate 10 technical interview questions.

Resume:
{context}
"""

    # -------------------------
    # Skill Suggestions
    # -------------------------

    elif "skill" in user_input.lower():

        prompt = f"""
Based on the resume below:

1. Suggest missing skills
2. Suggest trending technologies
3. Suggest a learning roadmap

Resume:
{context}
"""

    # -------------------------
    # General Resume Questions
    # -------------------------

    else:

        prompt = f"""
Answer ONLY using the resume information below.

If the answer is not present,
say:

I could not find that information in the resume.

Resume:
{context}

Question:
{user_input}

Answer:
"""

    print("\nAI is thinking...\n")

    response = llm.invoke(prompt)

    print(f"AI: {response.content}\n")