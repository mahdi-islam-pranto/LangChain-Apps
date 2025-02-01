from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

llm_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    # max_completion_tokens=100,
)

system_message = SystemMessage("You are a geography expert AI assistant.")
conversation_list = []
conversation_list.append(system_message)

while True:
    # take input from the user as a human message
    human_message = HumanMessage(input("You: "))
    if human_message.content == 'exit':
        break
    conversation_list.append(human_message)
    # send the prompt to the llm model
    response = llm_model.invoke(conversation_list)
    # create an AI message from the response
    response_message = AIMessage(response.content)
    conversation_list.append(response_message)
    # print the AI response
    print(f'AI: {response_message.content}')


print("Conversation Ended by User")
print("Conversation History:")
for message in conversation_list:
    print(message.content)
    


