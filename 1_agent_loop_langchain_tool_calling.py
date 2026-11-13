from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
load_dotenv()

MODEL = 'qwen3:1.7b'
# MODEL = 'gpt-6-luna'
MODEL_PROVIDER = 'ollama'
# MODEL_PROVIDER = 'openai'
MAX_ITER = 10

@tool
def get_product_price(product:str)->float:
    '''
    Look up porce of a product in the catalog.
    args: 
        product: the name of the product to look up the price for
    '''
    print(f'>>>Executing get_product_price on {product}')
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str)->float:
    '''
    Apply a discount tier to a price and return the final price
    args:
        price: price of the product to apply discount on
        discount tier: discount tier of the user
    '''
    print(f'>>>Executing apply_discount(price={price}, discount_tier={discount_tier})')
    discount_percentages = {'bronze': 5, 'silver': 10, 'gold': 15}
    discount = discount_percentages.get(discount_tier, 0)
    discounted_price = round(price * ( 1-discount / 100), 2)
    return discounted_price

@traceable(name='LangChain Agent Loop')
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(
        model=MODEL,
        model_provider=MODEL_PROVIDER
    )   
    llm_with_tools = llm.bind_tools(tools)
    print(f'Question: {question}\n')
    print('=' * 60)

    messages = [
        SystemMessage(
            content='''
                You are a helpful shop assistant
                You have access to two tools: 1. get_product_price  2. apply_discount

                Strict Rules:
                1. USE get_product_price to get the price of the product
                2. USE apply_discount to get the discounted price
                3. Only call apply_discount AFTER you have received a price from get_product_price - do NOT pass a made-up number.
                4. NEVER calculate the discount yourself, use the apply_discount tool.
                5. If the user ask you for anything other than the price of a product, politely reject the query and justify you only help with product price lookup and discount calculation 
            '''
        ),
        HumanMessage(
            content=question
        )
    ]
    for iteration in range(1, MAX_ITER+1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(input=messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.text}")
            return ai_message.content

        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            tool_call_id = tool_call.get('id')
            print(f'[Tool Selected]: {tool_name} with arguments {tool_args}')
        
            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError('Tool {tool_name} not exist')
            observation = tool_to_use.invoke(tool_args)
            print(f"[Tool Result]: {observation}")

            messages.append(ai_message)
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

def main():
    question = 'What is my discounted price for a laptop? My discount tier is bronze'
    run_agent(question=question)

if __name__ == '__main__':
    main()

