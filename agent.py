from langchain_ollama import ChatOllama
from datetime import datetime

# Load Local LLM
llm = ChatOllama(model="llama3.2:3b")


# Time Tool
def get_current_time():
    return datetime.now().strftime("%H:%M:%S")


# Calculator Tool
def calculate(expression):
    try:
        return eval(expression)
    except Exception:
        return "Invalid expression"


print("AI Agent Started!")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    elif "time" in user_input.lower():
        print("AI:", get_current_time())

    elif "calculate" in user_input.lower():
        expression = user_input.lower().replace("calculate", "").strip()
        print("AI:", calculate(expression))

    else:
        response = llm.invoke(user_input)
        print("AI:", response.content)

    print()