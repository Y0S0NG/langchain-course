from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from ollama import chat

load_dotenv()

MODEL = "qwen3:1.7b"
# MODEL = 'gpt-6-luna'
MODEL_PROVIDER = "ollama"
# MODEL_PROVIDER = 'openai'
MAX_ITER = 10


@traceable(name="tool_call: get_product_price")
def get_product_price(product: str) -> float:
    """
    Look up porce of a product in the catalog.
    Args:
        product: the name of the product to look up the price for

    Returns:
        price of the product
    """
    print(f">>>Executing get_product_price on {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price
    args:
        price: price of the product to apply discount on
        discount tier: discount tier of the user

    Returns
    """
    print(f">>>Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier, 0)
    discounted_price = round(price * (1 - discount / 100), 2)
    return discounted_price


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up porce of a product in the catalog.",
            "parameters": {
                "type": "object",
                "required": ["product"],
                "properties": {
                    "product": {"type": "string", "description": "name of a product"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and get the final price.",
            "parameters": {
                "type": "object",
                "required": ["price", "discount_tier"],
                "properties": {
                    "price": {
                        "type": "float", 
                        "description": "price of a product"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "discount_tier of a user",
                    },
                },
            },
        },
    },
]

@traceable(name='Ollama Chat', run_type='llm')
def ollama_chat_trace(messages):
    return chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    tools_dict = {'get_product_price': get_product_price, 'apply_discount': apply_discount}
    

    messages = [
        {'role': 'system', 
         'content':"""
                You are a helpful shop assistant
                You have access to two tools: 1. get_product_price  2. apply_discount

                Strict Rules:
                1. USE get_product_price to get the price of the product
                2. USE apply_discount to get the discounted price
                3. Only call apply_discount AFTER you have received a price from get_product_price - do NOT pass a made-up number.
                4. NEVER calculate the discount yourself, use the apply_discount tool.
                5. If the user ask you for anything other than the price of a product, politely reject the query and justify you only help with product price lookup and discount calculation 
            """},
        {'role': 'user', 'content': question},
    ]
    for iteration in range(1, MAX_ITER + 1):
        print(f"\n--- Iteration {iteration} ---")
        response = ollama_chat_trace(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        for tool_call in tool_calls:
            tool_to_use = tools_dict[tool_call.function.name]
            tool_args = tool_call.function.arguments
            
            observation = tool_to_use(**tool_args)
            print(f"[Tool Result]: {observation}")

            messages.append(ai_message)
            messages.append(
            {'role': 'tool', 'content': str(observation)}
            )


def main():
    question = "What is my discounted price for a laptop? My discount tier is bronze"
    run_agent(question=question)


if __name__ == "__main__":
    main()
