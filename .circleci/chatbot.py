def quick_fix_llm_json(llm_output: str):
    """
    Quick fix for common LLM JSON issues.
    """
    
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
