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
