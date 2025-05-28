def json_parse(llm_output: str):
    """
    Parse JSON from LLM output, handling various formatting issues.
    
    Args:
        llm_output: Raw string output from LLM that may contain JSON
        
    Returns:
        Parsed JSON object (dict/list)
    """
    # Remove BOM and strip whitespace
    text = llm_output.replace('\ufeff', '').strip()
    
    # Remove markdown code block markers
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    
    # Extract JSON object/array (find outermost braces/brackets)
    # First try to find object
    obj_start = text.find('{')
    obj_end = text.rfind('}')
    
    # Then try to find array
    arr_start = text.find('[')
    arr_end = text.rfind(']')
    
    # Determine which is the outermost structure
    if obj_start != -1 and obj_end != -1:
        if arr_start != -1 and arr_end != -1:
            # Both found, use the outermost
            if obj_start < arr_start:
                text = text[obj_start:obj_end+1]
            else:
                text = text[arr_start:arr_end+1]
        else:
            # Only object found
            text = text[obj_start:obj_end+1]
    elif arr_start != -1 and arr_end != -1:
        # Only array found
        text = text[arr_start:arr_end+1]
    
    # Fix smart quotes - THIS WAS THE BUG!
    # Replace various quote types with standard quotes
    text = text.replace('"', '"')  # Left smart quote
    text = text.replace('"', '"')  # Right smart quote
    text = text.replace(''', "'")  # Left smart single quote
    text = text.replace(''', "'")  # Right smart single quote
    text = text.replace('„', '"')  # German/Polish quote
    text = text.replace('"', '"')  # Another type of smart quote
    
    # Fix common JSON issues
    # Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Try to parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # If it still fails, try some more aggressive fixes
        
        # Fix unescaped newlines in strings
        text = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: m.group(1).replace('\n', '\\n'), text)
        
        # Fix single quotes (convert to double quotes carefully)
        # This is risky but sometimes necessary
        # Only do this if the JSON parse failed
        try:
            # Try parsing with single quotes replaced
            temp_text = re.sub(r"'([^']*)'", r'"\1"', text)
            return json.loads(temp_text)
        except:
            pass
        
        # Re-raise the original error if all fixes fail
        raise e
