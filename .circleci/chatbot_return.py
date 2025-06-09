import asyncio
from pathlib import Path
import os
import streamlit as st
from streamlitapp.utils import loadcss, clear_directory
from ai.chatbot.horizon_dev.horizon import horizon_main
from ai.chatbot.prompts.prompts import get_gripai_agent_prompt
from ai.chatbot.tools.tool_registry import get_tool_registry
from ai.chatbot.Json_output_parser.Json_parser import json_parse
from ai.chatbot.Json_output_parser.streamlit_content_formatter import display_mixed_content

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

if "file_info" not in st.session_state:
   st.session_state.file_info = None
if "directory_cleaned" not in st.session_state:
    st.session_state.directory_cleaned = False

if not st.session_state.directory_cleaned:
    default_dir = Path(__file__).parent / "ai/chatbot/data/temp"
    clear_directory(default_dir)
    st.session_state.directory_cleaned = True  

# Initialize session state for messages if not already initialized
if "messages" not in st.session_state:
    tool_registry = get_tool_registry()
    system_prompt = get_gripai_agent_prompt(tool_registry=tool_registry)
    st.session_state.messages = [{"role": "system", "content": system_prompt},
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
                    "type": "application/pdf"}],
            file_info = st.session_state.file_info))

        messages = response.get("messages", [])
        new_message =  messages[-1]["content"] if messages else "No response received."

        st.session_state.file_info = response.get("file_info")
        # print(response)

        # Append the assistant's response to the messages list
        st.session_state.messages.append({"role": "assistant", "content": new_message})

# if st.session_state.messages:
#   print(st.session_state.messages[1:])

# Display all chat messages from the session state
for message in st.session_state.messages:
    # print(message)
    if message["role"]!= "system":
      role = message["role"]
      if role == "assistant":
          # Try to access message["content"]["content"], fall back to message["content"]
          content = None
          try:
             content = json_parse(message["content"])["content"]
          except:
             print("Not parsable")
             content = message["content"]
          # print(type(content))
      else:
          content = message["content"]
      font_weight = "bold" if role == "assistant" else "normal"
      avatar_path = str(Path(__file__).parent / "streamlitapp/static/images/chatbot.png") if role == "assistant" else str(Path(__file__).parent / "streamlitapp/static/images/user.png")


      # Use markdown with HTML styling to set the font
      # st.markdown(f'<p style="font-family:ElevanceSans; font-weight:{font_weight};">{content}</p>', unsafe_allow_html=True)
      message = st.chat_message(role, avatar=avatar_path)
      display_mixed_content(content,message)


print(st.session_state.file_info)
if st.session_state.file_info:
        file_info = st.session_state.file_info
        filepath = file_info["filepath"]
        
        if os.path.exists(filepath):
            # Create download button
            with open(filepath, "rb") as file:
                st.download_button(
                    label=f"📥 Download {file_info['filename']}",
                    data=file.read(),
                    file_name=file_info["filename"],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


# from pathlib import Path
# from streamlitapp.utils import resize_gif

# resize_gif(Path(__file__).parent / "streamlitapp/static/images/carelon-logo.gif", 
#              Path(__file__).parent / "streamlitapp/static/images/carelon-logo1.gif", (45, 44))
