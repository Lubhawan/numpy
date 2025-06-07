def online_completions_llm_node(state: GraphState) -> GraphState:
    user_prompt = state.get("user_query", "what is 2+2 ?")
    if not user_prompt:
        raise ValueError("No prompt provided for processing.")


    files = state.get("files", [])
    if not files:
        raise ValueError("No files provided for processing.")

    chat_history = state.get("messages",[])
    # print("chat_history",chat_history)


    # payload = {"prompt": user_prompt}
    # payload = {'role':"user","content":user_prompt}
    payload = {"messages": chat_history, "stream": False}

    # print(payload)
    # endpoint = "/v2/document/online/completions"

    # endpoint = "/v2/document/chats"

    endpoint = "/v2/text/chats"

    
    file_tuples_list = []
    for file in files:
        file_path = Path(file["path"])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file['filename']}")
        with file_path.open('rb') as f:
            file_tuples_list.append(('files', (file["filename"], f.read(), file["type"])))

    # print("files", file_tuples_list)
    output=None
    if endpoint == '/v2/document/online/completions':
        llm = OnlineCompletionsLLM()
        output = llm._call(payload=payload,messages=[], files=file_tuples_list, endpoint=endpoint, stream=False)
    if endpoint == "/v2/document/chats":
        llm = OnlineChatCompletionsLLM()
        output = llm._call(payload=payload,messages=chat_history, files=file_tuples_list, endpoint=endpoint, stream=False)
    if endpoint == "/v2/text/chats":
        llm = TextChatCompletionsLLM()
        # print(system_prompt)
        output = llm._call(payload=payload, endpoint=endpoint, stream=False)
        # print(output)
        # print("output from llm",output["message"]["content"])
        # output=json.loads(output["message"])
        # output =[json.loads(output["message"]["content"])]
        # print(output)
    if not output:
        raise ValueError("No output received from the LLM.")
    
    return {**state, "messages": chat_history + [output]}



from ai.chatbot.graphs.GraphState import GraphState
from ai.chatbot.tools.tool_mapping import get_tool_mapping
from ai.chatbot.Json_output_parser.Json_parser import json_parse

def tool_execution_node(state:GraphState):
    agentOutput = json_parse(state["messages"][-1]['content'])
    tool_name = agentOutput["tool"]
    tool_input = agentOutput["tool_input"]
    use_case = agentOutput["use_case"]
    columns_to_extract = agentOutput["columns_to_extract"]
    function_name = get_tool_mapping(tool_name)
    if function_name is None:
        raise ValueError(f"Tool '{tool_name}' not found in tool mapping")
    print("tool_name",tool_name)
    print("tool_input",tool_input)
    print("use_case",use_case)
    print("function_name",function_name)
    results, file_info = function_name(columns_to_extract,**tool_input)

    # new_tool_message = {
    #     "role":"assistant",
    #     "content":f"[Tool Executed] Results of tool: {results}"
    # }
    # # print(new_tool_message)

    # return {"messages":state["messages"]+[new_tool_message]}

    new_tool_message = {
        'role': 'assistant',
        'content': f'''{{
            "type": "tool_results",
            "content": {results}"
        }}'''
    }
    # {'role': 'assistant', 'content': '```json\n{\n    "type": "tool_call",\n    "use_case": "use_case_1",\n    "columns_to_extract": [\n        "mixer_key",\n        "claim_type",\n        "company_code",\n        "mixer_description"\n    ],\n    "tool": "get_api_level_one_data",\n    "tool_input": {\n        "claim_type": null,\n        "member_product_code": null,\n        "variance": "CN00",\n        "company_code": null\n    },\n    "thought": "The query explicitly requests data for a specific variance \'CN00\'. This matches use_case_1, which allows searching by variance. Confidence level is high due to the exact match with the \'variance\' parameter."\n}\n```'}
    # print(new_tool_message)

    return {**state, "messages":state["messages"]+[new_tool_message], "file_info": file_info}
