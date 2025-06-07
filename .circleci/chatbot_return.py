# In your LangGraph node/tool
def excel_generation_tool(state):
    # Generate your Excel file
    df = create_your_dataframe()
    
    # Save with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_report_{timestamp}.xlsx"
    filepath = f"temp/{filename}"
    
    df.to_excel(filepath, index=False)
    
    # Return file info in state
    return {
        "response": "Analysis complete! Excel report generated.",
        "file_info": {
            "filename": filename,
            "filepath": filepath,
            "file_type": "excel"
        }
    }

# Your LangGraph graph should return this info
def run_langgraph_flow(messages):
    result = graph.invoke({"messages": messages})
    return {
        "response": result.get("response", ""),
        "file_info": result.get("file_info", None)
    }


import streamlit as st
import os

# Your existing Streamlit code
if user_input:
    # Call LangGraph
    result = run_langgraph_flow(st.session_state.messages)
    
    # Display response
    st.write(result["response"])
    
    # Check if file was generated
    if result.get("file_info"):
        file_info = result["file_info"]
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
            
            # Optional: Clean up temp file after some time
            # os.remove(filepath)  # Be careful with this
