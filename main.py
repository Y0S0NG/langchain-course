from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


load_dotenv()

class Source(BaseModel):
    '''
    Schema for a source used by the agent
    '''
    url:str = Field(description='The URL of the source')

class AgentResponse(BaseModel):
    '''
    Schema for agent response with answer and sources
    '''
    answer:str = Field(description='Agent answer to the query')
    sources:list[Source] = Field(default_factory=list, description='The list of sources used to generate the answer')


llm = ChatOpenAI(model='gpt-6-luna')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    response = agent.invoke({'messages':HumanMessage(content="Find me a good Japanese restaurant in Bay Area, with its link on Yelp")})
    print(response)

if __name__ == "__main__":
    main()
