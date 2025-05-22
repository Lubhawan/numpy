from typing import Dict, List, Any, Optional, TypedDict, Literal
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation


# Define our state
class ChatState(TypedDict):
    messages: List[Any]
    use_case: Optional[str]
    use_case_metadata: Optional[Dict[str, Any]]  # Stores confidence and reasoning for use case identification
    parameters: Optional[Dict[str, Any]]
    extraction_details: Optional[Dict[str, Any]]  # Stores confidence and other extraction metadata
    api_level: Optional[Literal["level_one", "level_two"]]
    api_response: Optional[Dict[str, Any]]

# Define the use cases and their required parameters
USE_CASES = {
    "use_case_1": {
        "description": "Check account balance",
        "required_parameters": ["account_number"],
        "parameter_descriptions": {
            "account_number": "The account number to check balance for"
        },
        "parameter_examples": {
            "account_number": ["12345678", "87654321"]
        },
        "trigger_phrases": ["balance", "how much money", "account balance", "check balance"],
        "example_inputs": ["What's my balance?", "How much money do I have in account 12345678?"],
        "api_level": "level_one"
    },
    "use_case_2": {
        "description": "Transfer money between accounts",
        "required_parameters": ["source_account", "destination_account", "amount"],
        "parameter_descriptions": {
            "source_account": "The account to transfer from",
            "destination_account": "The account to transfer to",
            "amount": "The amount to transfer"
        },
        "parameter_examples": {
            "source_account": ["12345678", "87654321"],
            "destination_account": ["87654321", "12345678"],
            "amount": ["100", "1000", "5.50"]
        },
        "trigger_phrases": ["transfer", "send money", "move funds"],
        "example_inputs": ["Transfer $100 from account 12345678 to 87654321", "Send $50 to account 87654321"],
        "api_level": "level_two"
    },
    # Add the remaining 12 use cases with similar structure
    "use_case_3": {
        "description": "Check transaction history",
        "required_parameters": ["account_number", "date_range"],
        "parameter_descriptions": {
            "account_number": "The account number to check transactions for",
            "date_range": "The date range for transactions (e.g., 'last week', 'past 30 days')"
        },
        "parameter_examples": {
            "account_number": ["12345678", "87654321"],
            "date_range": ["last week", "past 30 days", "yesterday", "last month"]
        },
        "trigger_phrases": ["transactions", "history", "activity", "statement"],
        "example_inputs": ["Show me transactions for account 12345678 from last week", "What's my account activity?"],
        "api_level": "level_two"
    }
    # Continue with use_cases 4-14
}Description of parameter 2"
        },
        "parameter_examples": {
            "param1": ["example1", "example2"],
            "param2": ["example3", "example4"]
        },
        "api_level": "level_one"
    },
    "use_case_2": {
        "description": "Description of use case 2",
        "required_parameters": ["param3", "param4"],
        "parameter_descriptions": {
            "param3": "Description of parameter 3",
            "param4": "Description of parameter 4"
        },
        "parameter_examples": {
            "param3": ["example5", "example6"],
            "param4": ["example7", "example8"]
        },
        "api_level": "level_two"
    },
    # Add all 14 use cases here
}

# Function to identify the use case from user input
def identify_use_case(state: ChatState) -> ChatState:
    messages = state["messages"]
    
    # Get the last message from the user
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state
    
    # Use LLM to identify the use case
    use_case_identifier = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create a more structured representation of use cases for the LLM
    use_case_info = {}
    for case_id, case_data in USE_CASES.items():
        use_case_info[case_id] = {
            "description": case_data["description"],
            "trigger_phrases": case_data.get("trigger_phrases", []),
            "example_inputs": case_data.get("example_inputs", [])
        }
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        You are a use case identifier for a specialized service chatbot. Your task is to analyze the user's message and identify which of the following use cases it most closely matches:
        
        {json.dumps(use_case_info, indent=2)}
        
        Guidelines for identification:
        1. Look for specific keywords, phrases, or intents that match a use case
        2. Consider synonyms and alternative phrasings
        3. If the user's message could match multiple use cases, choose the one with the highest confidence
        4. If no use case matches with at least 70% confidence, return "unknown"
        
        Return a JSON with the following structure:
        {
            "use_case": "[use_case_id]",
            "confidence": [0-100],
            "reasoning": "[brief explanation of why this use case was chosen]"
        }
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | use_case_identifier | JsonOutputParser()
    
    result = chain.invoke({"messages": messages})
    
    # Update state with identified use case
    return {
        **state,
        "use_case": result.get("use_case"),
        "use_case_metadata": {
            "confidence": result.get("confidence", 0),
            "reasoning": result.get("reasoning", "")
        }
    }

# Function to extract parameters for the identified use case
def extract_parameters(state: ChatState) -> ChatState:
    if not state.get("use_case"):
        return state
    
    messages = state["messages"]
    use_case = state["use_case"]
    
    # Get required parameters for this use case along with their descriptions and examples
    use_case_info = USE_CASES[use_case]
    required_params = use_case_info["required_parameters"]
    param_descriptions = use_case_info.get("parameter_descriptions", {})
    param_examples = use_case_info.get("parameter_examples", {})
    
    # Create a more detailed description for each parameter to help the LLM
    param_details = {}
    for param in required_params:
        param_details[param] = {
            "description": param_descriptions.get(param, f"Value for {param}"),
            "examples": param_examples.get(param, []),
            "required": True
        }
    
    # Use LLM to extract parameters with enhanced prompt
    parameter_extractor = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        You are a parameter extraction specialist. Your task is to carefully extract the following parameters from the user's message for use case '{use_case}':
        
        {json.dumps(param_details, indent=2)}
        
        Guidelines for extraction:
        1. Look for exact values in the user's message that match each parameter
        2. Consider synonyms and alternative phrasings
        3. For dates, recognize various formats (e.g., "tomorrow", "next Monday", "05/21/2025")
        4. For numerical values, extract both digits and written numbers
        5. Be precise - do not guess values that aren't clearly stated
        
        Return a JSON with all extracted parameters and their values. Include a confidence score (0-100) for each extraction.
        If a parameter is not found in the message, set its value to null and confidence to 0.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | parameter_extractor | JsonOutputParser()
    
    # Get the latest user message
    user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
    latest_user_message = user_messages[-1] if user_messages else ""
    
    # Invoke the chain with all messages for context but highlight the latest message
    extraction_context = {
        "messages": messages,
        "latest_message": latest_user_message
    }
    
    result = chain.invoke({"messages": messages})
    
    # Extract just the parameter values from the result
    # (removing confidence scores for the API call)
    parameters = {}
    for param, data in result.items():
        if isinstance(data, dict) and "value" in data:
            parameters[param] = data["value"]
        else:
            parameters[param] = data
    
    # Update state with extracted parameters and API level
    return {
        **state,
        "parameters": parameters,
        "extraction_details": result,  # Store full extraction details including confidence
        "api_level": USE_CASES[use_case]["api_level"]
    }

# Function to determine if we need more information from the user
def need_more_info(state: ChatState) -> Literal["ask_user", "call_api"]:
    if not state.get("use_case") or state.get("use_case") == "unknown":
        return "ask_user"
    
    use_case = state["use_case"]
    parameters = state.get("parameters", {})
    extraction_details = state.get("extraction_details", {})
    required_params = USE_CASES[use_case]["required_parameters"]
    
    # Check if any required parameters are missing or have low confidence
    for param in required_params:
        # Check if parameter is missing
        if param not in parameters or parameters[param] is None:
            return "ask_user"
        
        # If we have confidence scores, check if confidence is too low
        if extraction_details and param in extraction_details:
            if isinstance(extraction_details[param], dict) and "confidence" in extraction_details[param]:
                confidence = extraction_details[param]["confidence"]
                # If confidence is below threshold, ask for clarification
                if confidence < 70:  # 70% confidence threshold
                    return "ask_user"
    
    return "call_api"

# Function to ask user for more information
def ask_for_more_info(state: ChatState) -> ChatState:
    use_case = state.get("use_case")
    parameters = state.get("parameters", {})
    extraction_details = state.get("extraction_details", {})
    
    if not use_case or use_case == "unknown":
        # We couldn't identify a use case, ask the user for clarification
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # Get list of available use cases for suggestions
        use_case_examples = {}
        for case_id, case_data in USE_CASES.items():
            use_case_examples[case_id] = case_data["description"]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            You are a helpful assistant. The system couldn't clearly identify what the user is trying to do.
            
            Available services include:
            {json.dumps(use_case_examples, indent=2)}
            
            Ask for clarification about what they're trying to do. Be conversational and helpful.
            Provide 2-3 examples of the types of requests you can help with. Don't list all possibilities.
            """),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        response = llm.invoke(prompt.format(messages=state["messages"]))
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=response.content)],
        }
    
    # We have a use case but missing or low confidence parameters
    required_params = USE_CASES[use_case]["required_parameters"]
    param_descriptions = USE_CASES[use_case].get("parameter_descriptions", {})
    
    # Collect missing or low confidence parameters
    params_to_ask = []
    for param in required_params:
        # Parameter is missing
        if param not in parameters or parameters[param] is None:
            params_to_ask.append({
                "name": param,
                "description": param_descriptions.get(param, f"Value for {param}"),
                "reason": "missing"
            })
        # Parameter has low confidence
        elif extraction_details and param in extraction_details:
            if isinstance(extraction_details[param], dict) and "confidence" in extraction_details[param]:
                confidence = extraction_details[param]["confidence"]
                if confidence < 70:  # 70% confidence threshold
                    params_to_ask.append({
                        "name": param,
                        "description": param_descriptions.get(param, f"Value for {param}"),
                        "reason": "low_confidence",
                        "current_value": parameters[param]
                    })
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        You are a helpful assistant having a conversation with a user.
        
        Based on the conversation so far, I need to collect the following information:
        {json.dumps(params_to_ask, indent=2)}
        
        For each missing parameter:
        - Ask for the information in a natural, conversational way
        - Explain briefly why this information is needed (in user terms, not technical)
        - If a parameter has low confidence, show the value you think they meant and ask for confirmation
        
        Keep your message friendly, brief, and focused. Ask about all needed parameters in a single message.
        Don't use technical parameter names in your question.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    response = llm.invoke(prompt.format(messages=state["messages"]))
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }

# Function to call the appropriate API
def call_api(state: ChatState) -> ChatState:
    api_level = state["api_level"]
    parameters = state["parameters"]
    use_case = state["use_case"]
    
    # Format parameters according to API requirements
    formatted_params = {}
    for key, value in parameters.items():
        if value is not None:  # Only include non-null parameters
            # Format transformation could be applied here if needed
            formatted_params[key] = value
    
    # Call level one or level two API based on the requirement
    try:
        if api_level == "level_one":
            # In production, replace with actual API call
            # Example:
            # response = requests.post(
            #     "https://api.example.com/level-one",
            #     json=formatted_params,
            #     headers={"Authorization": "Bearer YOUR_API_KEY"}
            # )
            # api_response = response.json()
            
            # Simulated API response
            api_response = {
                "status": "success",
                "data": {
                    "column1": f"Data for {formatted_params.get('param1', 'unknown')}",
                    "column2": f"Related info for {formatted_params.get('param2', 'unknown')}",
                    "timestamp": "2025-05-21T14:30:00Z"
                }
            }
        else:  # level_two
            # In production, replace with actual API call
            # Example:
            # response = requests.post(
            #     "https://api.example.com/level-two",
            #     json=formatted_params,
            #     headers={"Authorization": "Bearer YOUR_API_KEY"}
            # )
            # api_response = response.json()
            
            # Simulated API response with more detailed data
            api_response = {
                "status": "success",
                "data": {
                    "main_info": {
                        "column1": f"Detailed data for {formatted_params.get('param3', 'unknown')}",
                        "column2": f"Extended info for {formatted_params.get('param4', 'unknown')}"
                    },
                    "additional_details": {
                        "history": ["item1", "item2", "item3"],
                        "statistics": {
                            "count": 42,
                            "average": 3.14
                        }
                    },
                    "timestamp": "2025-05-21T14:30:00Z"
                }
            }
        
        return {
            **state,
            "api_response": api_response
        }
    except Exception as e:
        # Handle API errors
        return {
            **state,
            "api_response": {
                "status": "error",
                "message": f"Failed to retrieve data: {str(e)}",
                "error_code": "API_ERROR"
            }
        }

# Function to format the API response for the user
def format_response(state: ChatState) -> ChatState:
    api_response = state["api_response"]
    use_case = state["use_case"]
    parameters = state["parameters"]
    
    # Handle error responses
    if api_response.get("status") == "error":
        error_message = api_response.get("message", "An unknown error occurred.")
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        error_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""
            The API request for use case '{use_case}' failed with the following error:
            {error_message}
            
            Please format a friendly error message for the user that:
            1. Acknowledges the issue
            2. Explains what happened in non-technical terms
            3. Suggests a possible next step
            
            Do not mention API calls, status codes, or other technical details.
            """),
        ])
        
        error_response = llm.invoke(error_prompt)
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=error_response.content)],
        }
    
    # For successful responses, format the data nicely
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        You are a helpful assistant presenting information to a user. The system has retrieved the following data:
        {json.dumps(api_response, indent=2)}
        
        This data is for use case '{use_case}' with these parameters:
        {json.dumps(parameters, indent=2)}
        
        Format instructions:
        1. Present the information in a clear, conversational, and user-friendly way
        2. Use markdown formatting to organize the information (tables, bullet points, etc.)
        3. Highlight the most important information first
        4. Don't mention the technical details of the API, use case names, or parameter names
        5. If the data includes dates or timestamps, format them in a readable way
        6. If the data is tabular, present it as a markdown table
        
        Your response should feel like a natural conversation, not a data dump.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    response = llm.invoke(prompt.format(messages=state["messages"]))
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }

# Build the graph
def build_graph():
    # Initialize the graph
    graph = StateGraph(ChatState)
    
    # Add nodes
    graph.add_node("identify_use_case", identify_use_case)
    graph.add_node("extract_parameters", extract_parameters)
    graph.add_node("ask_for_more_info", ask_for_more_info)
    graph.add_node("call_api", call_api)
    graph.add_node("format_response", format_response)
    
    # Add edges
    graph.add_edge("identify_use_case", "extract_parameters")
    graph.add_conditional_edges(
        "extract_parameters",
        need_more_info,
        {
            "ask_user": "ask_for_more_info",
            "call_api": "call_api",
        },
    )
    graph.add_edge("ask_for_more_info", END)
    graph.add_edge("call_api", "format_response")
    graph.add_edge("format_response", END)
    
    # Set the entry point
    graph.set_entry_point("identify_use_case")
    
    return graph

# Create a runnable from the graph
graph = build_graph()
chain = graph.compile()

# Example usage
def process_user_message(user_message: str, chat_history: List[Dict[str, Any]] = None):
    if chat_history is None:
        chat_history = []
    
    # Convert chat history to the expected format
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    # Add the new user message
    messages.append(HumanMessage(content=user_message))
    
    # Initial state
    state = {
        "messages": messages,
        "use_case": None,
        "parameters": None,
        "api_level": None,
        "api_response": None,
    }
    
    # Run the graph
    result = chain.invoke(state)
    
    # Get the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    
    return "I'm sorry, something went wrong. Please try again."
