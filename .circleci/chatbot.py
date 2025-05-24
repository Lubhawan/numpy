import pandas as pd
import json
import re
import streamlit as st

def display_mixed_content(data):
    """Display mixed text and JSON content using streamlit"""
    
    # Try to detect if data contains JSON
    try:
        # Look for JSON patterns (objects starting with { or arrays with [)
        json_pattern = r'(\{.*?\}|\[.*?\])'
        json_matches = re.findall(json_pattern, data, re.DOTALL)
        
        if json_matches:
            # Split content into text and JSON parts
            parts = re.split(json_pattern, data, flags=re.DOTALL)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                # Try to parse as JSON
                if part.startswith(('{', '[')):
                    try:
                        json_data = json.loads(part)
                        
                        # Display as table if it's a list of objects
                        if isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
                            st.subheader("Table Data")
                            df = pd.DataFrame(json_data)
                            st.dataframe(df)
                        
                        # Display as JSON if it's a single object or other structure
                        elif isinstance(json_data, dict):
                            st.subheader("JSON Data")
                            st.json(json_data)
                        
                        else:
                            st.subheader("Raw JSON")
                            st.json(json_data)
                            
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as text
                        st.write(part)
                else:
                    # Display as regular text
                    st.write(part)
        else:
            # No JSON found, display as text
            st.write(data)
            
    except Exception as e:
        # Fallback: display as text
        st.write(data)
