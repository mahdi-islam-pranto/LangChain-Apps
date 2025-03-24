from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool


# Define the tool DuckDuckGo
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_tool", func=search.run, description="Searches the web for information"
    )

# define the wikipedia tool
wiki_api = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=200
)
wiki_tool = WikipediaQueryRun(api_wrapper= wiki_api)

