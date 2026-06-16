import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# -------------------------
# Embeddings
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Create / Load ChromaDB
# -------------------------

if not os.path.exists("./chroma_db"):

    print("Creating Vector Database for first time...")

    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )

    print("Vector Database Created Successfully!")

else:

    print("Loading Existing Vector Database...")

    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    print("Vector Database Loaded Successfully!")

# -------------------------
# LLM
# -------------------------

llm = ChatOllama(
    model="qwen3:4b"
)

# -------------------------
# Session Memory
# -------------------------

history = []

print("\nCareer Assistant v5 Ready!")
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
    # Resume Score
    # -------------------------

    elif "score" in user_input.lower():

        prompt = f"""
You are an expert resume reviewer.

Review the resume below.

Resume:
{context}

Provide:

1. Overall Resume Score out of 10

2. Scores for:
- Technical Skills
- Experience
- Projects
- Certifications
- Resume Structure

3. Top Strengths

4. Areas for Improvement

5. Recommendations
"""

    # -------------------------
    # Interview Questions
    # -------------------------

    elif "interview" in user_input.lower():

        prompt = f"""
Based on the resume below,
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
You are a senior technical recruiter and career coach.

Resume Information:
{context}

Job Description:
{job_description}

Perform a detailed analysis.

Provide:

1. Match Score (0-100)

2. Matching Skills
   - Skills already present in the resume

3. Critical Missing Skills
   - Skills mandatory for the role

4. Important Missing Skills
   - Skills that would significantly improve the candidate's chances

5. Learning Roadmap
   - What to learn first
   - What to learn next
   - Suggested learning order

6. Estimated Job Readiness
   - Immediate
   - 1 Month
   - 3 Months
   - 6 Months

7. Hiring Recommendation

Use bullet points and keep the response professional.
"""

    # -------------------------
    # Resume Q&A
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

    # -------------------------
    # Memory Context
    # -------------------------

    conversation_history = "\n".join(history)

    final_prompt = f"""
Previous Conversation:
{conversation_history}

Current Task:
{prompt}
"""

    print("\nAI is thinking...\n")

    response = llm.invoke(final_prompt)

    print(f"AI:\n{response.content}\n")

    # -------------------------
    # Store Memory
    # -------------------------

    history.append(f"User: {user_input}")
    history.append(f"AI: {response.content}")

    # Keep memory limited
    if len(history) > 20:
        history = history[-20:]