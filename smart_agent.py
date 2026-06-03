from langchain_ollama import ChatOllama
from datetime import datetime

# Load LLM
llm = ChatOllama(model="llama3.2:3b")

# Conversation Memory
history = []


# ----------------------------
# TOOLS
# ----------------------------

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")


def calculate(expression):
    try:
        return eval(expression)
    except Exception:
        return "Invalid expression"


# ----------------------------
# ROUTER
# ----------------------------

def select_tool(user_input):

    router_prompt = f"""
You are a routing assistant.

Available tools:

TIME
- Use ONLY when the user asks for the current time.

CALCULATOR
- Use ONLY for arithmetic calculations.
- Examples:
  - What is 25 * 47?
  - calculate 100 + 50

LLM
- Use for everything else.

Question:
{user_input}

Return ONLY one word:
TIME
CALCULATOR
LLM
"""

    response = llm.invoke(router_prompt)

    tool = response.content.strip().split()[0].upper()

    if tool not in ["TIME", "CALCULATOR", "LLM"]:
        tool = "LLM"

    return tool


# ----------------------------
# MAIN
# ----------------------------

print("Smart AI Agent Started!")
print("Type 'exit' to quit\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Save user message
    history.append(f"Human: {user_input}")

    # Decide tool
    tool = select_tool(user_input)

    print(f"[Selected Tool: {tool}]")

    # TIME TOOL
    if tool == "TIME":

        ai_response = get_current_time()

    # CALCULATOR TOOL
    elif tool == "CALCULATOR":

        expression = (
            user_input.lower()
            .replace("calculate", "")
            .replace("what is", "")
            .replace("?", "")
            .strip()
        )

        ai_response = str(calculate(expression))

    # LLM + MEMORY
    else:

        prompt = "\n".join(history)

        response = llm.invoke(prompt)

        ai_response = response.content

    # Save AI response
    history.append(f"AI: {ai_response}")

    print("\nAI:", ai_response)
    print()