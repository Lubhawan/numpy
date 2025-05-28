```json
{
    "type": "tool_call",
    "use_case": "use_case_1",
    "columns_to_extract": [
        "mixer_key",
        "claim_type",
        "company_code",
        "mixer_description"
    ],
    "tool": "get_api_level_one_data",
    "tool_input": {
        "claim_type": null,
        "member_product_code": null,
        "variance": "CN00",
        "company_code": null
    },
    "thought": "The query explicitly requests data for a specific variance 'CN00'. This matches use_case_1, which allows searching by variance. The confidence level is high due to the direct match with the 'variance' parameter."
}
```

import re, json

def json_parse(llm_output: str):
        
        text = llm_output.replace('\ufeff', '').strip()
        
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        text = text.replace('"', '"').replace('"', '"')
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            return json.loads(text)


import pandas as pd
import json
import re
import streamlit as st

def display_mixed_content(data, message):
    try:
        json_pattern = r'(\{.*?\}|\[.*?\])'
        json_matches = re.findall(json_pattern, data, re.DOTALL)
        
        if json_matches:
            parts = re.split(json_pattern, data, flags=re.DOTALL)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                if part.startswith(('{', '[')):
                    try:
                        json_data = json.loads(part)
                        
                        if isinstance(json_data, list) and json_data and isinstance(json_data[0], dict):
                            print("Hi")
                            st.subheader("Table Data")
                            df = pd.DataFrame(json_data)
                            message.dataframe(df)
                        
                        elif isinstance(json_data, dict):
                            print("hi1")
                            message.subheader("JSON Data")
                            message.json(json_data)
                        
                        else:
                            print("hi2")
                            message.subheader("Raw JSON")
                            message.json(json_data)
                            
                    except json.JSONDecodeError:
                        print("hi3")
                        message.write(part)
                else:
                    print("hi4")
                    message.write(part)
        else:
            print("no json found")
            message.write(data)
            
    except Exception as e:
        print("hi6")
        message.write(data)
