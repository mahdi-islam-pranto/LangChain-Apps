from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    # max_completion_tokens=100,
)

result = llm.invoke("What is the capital of Bangladesh?")

print(result.content)

