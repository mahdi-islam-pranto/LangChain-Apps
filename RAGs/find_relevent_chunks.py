import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
load_dotenv()


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# The path to the database directory
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
    )

# load the existing vector store with the embeddings
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function= embedding_model
    )

# Get the user query/promt text
user_query = "What are center solution in iHelpBD?"

# retrieve the most relevent chunks from the database for the user query
retriver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.5},  
)

relevennt_chunks = retriver.invoke(user_query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
print("chunk:1" +relevennt_chunks[0].page_content, "\n")
print("chunk:2" +relevennt_chunks[1].page_content, "\n")


# put the relevant chunks + user query into the LLM model openai and generate the answer 
modified_prompt = f'Here is the User Query: "  {user_query} + "\n" + "And Here are the relevant Information where you can find the ansewer" + "\n" + "do not use other info other than the additional information I provided to you." + "\n" + "Relevat infomations: " + "\n" + {relevennt_chunks[0].page_content} + "\n" + {relevennt_chunks[1].page_content}'

# create the chat model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    
)

# generate the answer
response = llm.invoke(modified_prompt)


print("\n--- Answer ---")
print(response.content)

