from langchain_openai import ChatOpenAI

# load the .env file for open ai api key
from dotenv import load_dotenv
load_dotenv()

# create an instance of the llm model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    # max_completion_tokens=100,
)

# send the prompt to the llm model
result = llm.invoke("What is the capital of Bangladesh?")

print(result.content)

