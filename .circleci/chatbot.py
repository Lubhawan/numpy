import asyncio
from pathlib import Path
import streamlit as st
from streamlitapp.utils import loadcss
from ai.horizon_dev.horizon import horizon_main

st.set_page_config(page_title="Chat with GRIP-BOT",
                   page_icon=None, 
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)


loadcss()


st.logo(str(Path(__file__).parent / "streamlitapp/static/images/elevance-logo.png"), size="large")

# Example of applying font style to st.write
cols = st.columns([0.28, 0.72])

with cols[0]:
  st.header("Chat with GRIP-BOT")
with cols[1]:
  st.markdown("<p></p>",  # noqa: F541
        unsafe_allow_html=True
    )
  st.image(str(Path(__file__).parent / "streamlitapp/static/images/carelon-logo.gif"))
  

st.subheader("You can ask about your mixer, and I will help you with it", divider=True)




# Initialize session state for messages if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assisstant. You reply to user query cheerfully"},
                                 {"role": "assistant", "content": "Hello! How can I assist you today?"}]

# Check if a user input is provided
if prompt := st.chat_input():
    # Append the user's message to the messages list
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simulate an AI response
    with st.spinner("Thinking..."):
        response = asyncio.run(horizon_main(
            thread_id="thread-1",
            messages=st.session_state.messages,
            files=[{"filename": "dog.pdf",
                    "path": str(Path(__file__).parent / "streamlitapp/static/files/dog.pdf"),
                    "type": "application/pdf"}]
        ))

        # Append the assistant's response to the messages list
        st.session_state.messages.append({"role": "assistant", "content": response})

# if st.session_state.messages:
#   print(st.session_state.messages)

# Display all chat messages from the session state
for message in st.session_state.messages:
    # print(message)
    role = message["role"]
    content = message["content"]
    font_weight = "bold" if role == "assistant" else "normal"
    avatar_path = str(Path(__file__).parent / "streamlitapp/static/images/chatbot.png") if role == "assistant" else str(Path(__file__).parent / "streamlitapp/static/images/user.png")


    # Use markdown with HTML styling to set the font
    # st.markdown(f'<p style="font-family:ElevanceSans; font-weight:{font_weight};">{content}</p>', unsafe_allow_html=True)
    message = st.chat_message(role, avatar=avatar_path)
    message.write(content)


# from pathlib import Path
# from streamlitapp.utils import resize_gif

# resize_gif(Path(__file__).parent / "streamlitapp/static/images/carelon-logo.gif", 
#              Path(__file__).parent / "streamlitapp/static/images/carelon-logo1.gif", (45, 44))


messages list is not getting passed to the below function in other file, what is the reason? The function name is horizon_main



from pathlib import Path
from typing import Annotated, TypedDict, List, Dict
# import asyncio


from langgraph.graph import add_messages
from langgraph.graph import StateGraph
# from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from ai.horizon_dev.horizon_classes import OnlineCompletionsLLM, OnlineChatCompletionsLLM, TextChatCompletionsLLM

load_dotenv()

memory = InMemorySaver()

class File(TypedDict):
   filename: str
   path: str
   type: str

# Define the structure of the state
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # defines how this state key should be updated (appending messages).
    messages: List[Dict[str, str]]
    prompt: str
    files: list[File]

# Define a LangGraph node that uses the custom LLM
# def online_completions_llm_node(state: State) -> State:
#     user_prompt = state.get("prompt", "Find the first and second words in the attached document")
#     if not user_prompt:
#         raise ValueError("No prompt provided for processing.")


#     files = state.get("files", [])
#     if not files:
#         raise ValueError("No files provided for processing.")


#     payload = {"prompt": user_prompt}
#     endpoint = "/v2/document/online/completions?qos=cheap"

    
#     file_tuples_list = []
#     for file in files:
#         file_path = Path(file["path"])
#         if not file_path.exists():
#             raise FileNotFoundError(f"File not found: {file['filename']}")
#         with file_path.open('rb') as f:
#             file_tuples_list.append(('files', (file["filename"], f.read(), file["type"])))

#     # print("files", file_tuples_list)

#     llm = OnlineCompletionsLLM()
#     output = llm._call(payload=payload, files=file_tuples_list, endpoint=endpoint, stream=False)
#     if not output:
#         raise ValueError("No output received from the LLM.")
    
#     return {"messages": output, "prompt": user_prompt, "files": files}

def online_completions_llm_node(state: State) -> State:
    
    messages = state.get("messages", [])
    payload = {"messages": messages, "stream": False}
    endpoint = "/v2/text/chats?qos=cheap"

    llm = TextChatCompletionsLLM()
    output = llm._call(payload=payload, endpoint=endpoint, stream=False)
    if not output:
        raise ValueError("No output received from the LLM.")
    
    return {"messages": messages + [output]}



graph_builder = StateGraph(State)

graph_builder.add_node("OnlineCompletionsLLMNode", online_completions_llm_node)

# Define the entry point and how to route between nodes
# graph_builder.add_edge(START, "OnlineCompletionsLLMNode")
# graph_builder.add_edge("OnlineCompletionsLLMNode", END)

graph_builder.set_entry_point("OnlineCompletionsLLMNode")
graph_builder.set_finish_point("OnlineCompletionsLLMNode")

# Compile the graph into a runnable object
graph = graph_builder.compile(checkpointer=memory)


async def horizon_main(thread_id: str = "thread-1",
                      messages: List[Dict[str, str]] = None,
                      files: List[File] = None) -> str:
    # Ensure messages are in dictionary format
    messages = messages or []
    print(messages)

    # Invoke the graph with messages as dictionaries
    response = await graph.ainvoke(
        {"messages": messages, "files": files},
        {"configurable": {"thread_id": thread_id}}
    )

    # Extract the latest assistant message content
    messages = response.get("messages", [])
    return messages[-1]["content"] if messages else "No response received."
