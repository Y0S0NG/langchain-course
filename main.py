from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()



llm = ChatOpenAI(model='gpt-6-luna')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    response = agent.invoke({'messages':HumanMessage(content="Give me three job in Bay area requiring Langchain on linkedin, with their links")})
    print(response)

if __name__ == "__main__":
    main()
