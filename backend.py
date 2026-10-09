from faiss import loader
from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph.message import add_messages
import sqlite3

from langgraph.graph.message import add_messages

# from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
# from langchain_core.tools import tool

import requests
import math
import os


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
# from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command


load_dotenv(override=True)


llm = ChatOpenAI()

# Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_rag_document(file_path):
    # "./DataSheet_ANO-L7012R_20250905_EN_173813.pdf"
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF document.

    Args:
        query (str): The user's question or search query used to retrieve PDF content.

    Returns:
        str: The relevant information retrieved from the PDF document.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information found in the PDF document."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}:\n"
            f"Source: {source}\n"
            f"Page: {page}\n" 
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be avalid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try: 
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "Alpha Vantage API key is not set in environment variables."
        }

    url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        response = requests.get(
            url, 
            params=params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        return {
            "error": f"Failed to fetch stock price: {error}"
        }


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Purchase a specified quantity of shares for a given stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" /anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve purchase of {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower().strip() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}",
            "symbol": symbol,
            "quantity": quantity,  
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase order cancelled for {quantity} shares of {symbol} by human.",
            "symbol": symbol,
            "quantity": quantity,  
        }
        



@tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather for a given location.
    """
    
    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return data.get("weather", [{}])[0].get("description", "Unknown")


# Make tool list
tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# State
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

# Nodes 1
# def chat_node(state: ChatState):

#     # take user query from state
#     messages = state['messages']
    
#     # process the messages (e.g., send to LLM for response)
#     response = llm_with_tools.invoke(messages)
    
#     # update the state with the new message
#     return {'messages': [response]}

def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n "
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex ."
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks for the current weather at a location.\n\n"
            "- Use `purchase_stock` when the user wants to buy shares of a stock.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiveing a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state['messages']
    ]

    response = llm_with_tools.invoke(messages)

    return {'messages': [response]}

# Nodes 2 - Tool Node
tool_node = ToolNode(tools) # Executes tool calls

conn = sqlite3.connect(database="chatebot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# # Add Nodes
# graph.add_node('chat_node',chat_node)

# # Add Edges
# graph.add_edge(START, 'chat_node')
# graph.add_edge('chat_node', END)

# graph structure
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

# If the LLm asked for a tool, go to ToolNode; else finsh
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpoint)


# Helper functions for Streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    
    return list(all_threads)


def resume_interrupts(result, config):
    while interrupts := result.get("__interrupt__", []):
        resume_values = {}

        for pending_interrupt in interrupts:
            print(f"HITL: {pending_interrupt.value}")
            decision = input("Your decision: ").strip().lower()
            resume_values[pending_interrupt.id] = decision

        result = chatbot.invoke(
            Command(resume=resume_values),
            config=config,
        )

    return result

