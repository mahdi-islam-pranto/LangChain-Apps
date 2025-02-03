import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# The path to the text document to be processed
file_path = os.path.join(current_dir, "documents", "ihelpBD_doc.txt")
# The directory where the Chroma vector store will be saved
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    # Read the text content from the file
    text_loader = TextLoader(file_path)
    document = text_loader.load()

    # Split the text into chunks/characters
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    document_chunks = text_splitter.split_documents(document)

    # Display information about the split documents
    print(f"Document split into {len(document_chunks)} chunks.")
    print(f"First chunk: {document_chunks[0]}")
    print(f"2nd chunk: {document_chunks[1]}")


else:
    print("Persistent directory & vector Storage alredy exists.")




