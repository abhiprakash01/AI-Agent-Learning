from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"""
You are a tool router.

Available tools:
TIME
CALCULATOR
LLM

Rules:
- If the question asks for current time, return TIME.
- If the question involves mathematical calculations, return CALCULATOR.
- Otherwise return LLM.

Question:
{user_input}

Return ONLY one word:
TIME
CALCULATOR
LLM
"""

    response = llm.invoke(prompt)

    print("Selected Tool:", response.content.strip())
    print()