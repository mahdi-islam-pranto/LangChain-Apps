from langchain_core.prompts import PromptTemplate 
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# basic template
template = "Write a {writing} about '{topic}' for class {grade}(school) students. Make it simple and easy to understand ad it is for school students. make it {length} words long."

# create a prompt template from the basic template
promt_template = PromptTemplate.from_template(template)


chat_promt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a school study assistant."),
    ("human", template),
    
])

final_chat_promt_template = chat_promt_template.format_messages(writing="paragraph", topic="My father", grade="5th", length="50")


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    
)

# print(final_chat_promt_template)

# send the prompt to the llm model
response =  llm.invoke(final_chat_promt_template)

print(f'AI : {response.content}')





