import pandas as pd
import json
import re

def display_mixed_content(data):
    """Display mixed text and JSON content using print statements"""
    
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
                            print("\n=== Table Data ===")
                            df = pd.DataFrame(json_data)
                            print(df.to_string(index=False))  # Pretty-print DataFrame
                        
                        # Display as JSON if it's a single object or other structure
                        elif isinstance(json_data, dict):
                            print("\n=== JSON Data ===")
                            print(json.dumps(json_data, indent=2))  # Pretty-print JSON
                        
                        else:
                            print("\n=== Raw JSON ===")
                            print(json.dumps(json_data, indent=2))  # Pretty-print JSON
                            
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as text
                        print(part)
                else:
                    # Display as regular text
                    print(part)
        else:
            # No JSON found, display as text
            print(data)
            
    except Exception as e:
        # Fallback: display as text
        print(data)


import re, json

def json_parse(llm_output: str):
        
        # Remove BOM and normalize
        text = llm_output.replace('\ufeff', '').strip()
        
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Extract JSON portion (find first { to last })
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        # Fix smart quotes
        text = text.replace('"', '"').replace('"', '"')
        
        # Try to parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fix trailing commas and try again
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            return json.loads(text)
