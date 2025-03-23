from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Define the schema for the structured output
class ResponseStructure(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Define the chat template
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a research assistant that will generate a short research paper.
         Answer the user query and use necessary sources and tools.
         Wrap the response in this structured format and no other text \n {formatted_output}.
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{user_query}"),
        ("placeholder", "{initial_agent}"),
    ]
)

