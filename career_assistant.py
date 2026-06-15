from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# -------------------------
# Load Resume
# -------------------------

loader = PyPDFLoader("resume.pdf")
documents = loader.load()

# -------------------------
# Split Resume
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# -------------------------
# Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Vector Database
# -------------------------

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("Vector Database Created Successfully!")

# -------------------------
# LLM
# -------------------------

llm = ChatOllama(
    model="qwen3:4b"
)

print("\nCareer Assistant Ready!")
print("Type 'exit' to quit.\n")

# -------------------------
# Main Loop
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

Analyze the resume below.

Resume:
{context}

Provide:

1. Top 3 Strengths
2. Top 3 Areas for Improvement
3. Top 3 Skills to Learn Next
"""

    # -------------------------
    # Interview Questions
    # -------------------------

    elif "interview" in user_input.lower():

        prompt = f"""
Based on the resume below, generate 10 technical interview questions.

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
    # JD Matcher
    # -------------------------

    elif "match" in user_input.lower():

        print("\nPaste Job Description")
        print("Type END on a new line when finished:\n")

        lines = []

        while True:

            line = input()

            if line.upper() == "END":
                break

            lines.append(line)

        job_description = "\n".join(lines)

        prompt = f"""
You are an expert technical recruiter.

Resume Information:
{context}

Job Description:
{job_description}

Provide:

1. Match Score (0-100)
2. Strong Matching Skills
3. Missing Skills
4. Improvement Suggestions
5. Hiring Recommendation

Keep the answer concise and professional.
"""

    # -------------------------
    # General Resume Q&A
    # -------------------------

    else:

        prompt = f"""
Answer ONLY using the resume information below.

If the answer is not available in the resume,
respond with:

I could not find that information in the resume.

Resume:
{context}

Question:
{user_input}

Answer:
"""

    print("\nAI is thinking...\n")

    response = llm.invoke(prompt)

    print(f"AI:\n{response.content}\n")