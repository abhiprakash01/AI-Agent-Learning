from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b")

history = []

print("AI Assistant Started!")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    history.append(f"Human: {user_input}")

    prompt = "\n".join(history)

    response = llm.invoke(prompt)

    ai_response = response.content

    history.append(f"AI: {ai_response}")

    print("\nAI:", ai_response)
    print()