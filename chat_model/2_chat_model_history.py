from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    # max_completion_tokens=100,
)

AI_role = "geography"
last_chat = "What is the biggest river of this country?"

# previous_chat_ 
promts = ChatPromptTemplate.from_messages([
    ("system", "You are a {AI_role} expert."),
    ("human", "What is the capital of Bangladesh?"),
    ("system", "The capital of Bangladesh is Dhaka."),
    ("human", last_chat),
])

promts = promts.format_messages(AI_role=AI_role)

# send the prompt to the llm model
result = llm.invoke(promts)

print(result.content)

