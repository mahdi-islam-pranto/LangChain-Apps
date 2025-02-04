import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
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
user_query = "what is the call center solution of ihelpBD?"

# retrieve the most relevent chunks from the database for the user query
retriver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": 0.7},  
)

relevennt_chunks = retriver.invoke(user_query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
print("chunk:1" + relevennt_chunks[0], "\n")
print("chunk:2" +relevennt_chunks[1], "\n")


