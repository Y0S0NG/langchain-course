from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model='qwen3:1.7b', reasoning=True)

def main():
    query = input("User: ")
    response = llm.invoke(input=query)
    print(response.content)

if __name__ == "__main__":
    main()
