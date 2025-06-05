import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import base64
from io import BytesIO
import re

# Define the state structure
class ChatDataInsightState(TypedDict):
    messages: List[Any]  # Chat history
    file_path: Optional[str]
    raw_data: Optional[pd.DataFrame]
    data_info: Optional[Dict[str, Any]]  # Basic info about the data
    current_query: str
    query_type: Optional[str]  # "visualization", "analysis", "question"
    generated_code: Optional[str]
    execution_result: Optional[Any]
    visualization: Optional[str]  # Base64 encoded image
    insights: Optional[str]
    error: Optional[str]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Node functions
def load_and_profile_data(state: ChatDataInsightState) -> ChatDataInsightState:
    """Load and create a basic profile of the data"""
    try:
        file_path = state.get("file_path")
        if not file_path:
            return state
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            state["error"] = f"Unsupported file format: {file_path}"
            return state
        
        state["raw_data"] = df
        
        # Create data info for LLM context
        data_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        state["data_info"] = data_info
        
    except Exception as e:
        state["error"] = f"Error loading data: {str(e)}"
    
    return state

def classify_query(state: ChatDataInsightState) -> ChatDataInsightState:
    """Use LLM to classify the user's query type"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analysis assistant. Classify the user's query into one of these categories:
        - "visualization": User wants to create charts, plots, or visual representations
        - "analysis": User wants statistical analysis, calculations, or data insights
        - "question": User is asking a question about the data
        
        Respond with only the category name."""),
        ("human", "{query}")
    ])
    
    response = llm.invoke(prompt.format_messages(query=state["current_query"]))
    state["query_type"] = response.content.strip().lower()
    
    return state

def generate_code(state: ChatDataInsightState) -> ChatDataInsightState:
    """Use LLM to generate Python code based on the query"""
    
    data_info = state.get("data_info", {})
    query_type = state.get("query_type", "analysis")
    
    system_prompt = f"""You are a data analysis expert. Generate Python code to answer the user's query.
    
    Available data:
    - DataFrame variable: df
    - Shape: {data_info.get('shape', 'Unknown')}
    - Columns: {data_info.get('columns', [])}
    - Data types: {json.dumps(data_info.get('dtypes', {}), indent=2)}
    - Numeric columns: {data_info.get('numeric_columns', [])}
    - Categorical columns: {data_info.get('categorical_columns', [])}
    
    Instructions:
    1. For visualizations: Use matplotlib or seaborn. Always include plt.figure() and plt.tight_layout()
    2. For analysis: Calculate and print results clearly
    3. For questions: Extract and format the answer
    4. Always handle potential errors gracefully
    5. Make visualizations attractive with proper labels, titles, and colors
    6. DO NOT include data loading - assume 'df' exists
    7. For visualizations, DO NOT use plt.show() - the code will handle saving
    
    Return only the Python code, no explanations."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["current_query"])
    ]
    
    # Add conversation context
    for msg in state.get("messages", [])[-4:]:  # Last 4 messages for context
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))
    
    response = llm.invoke(messages)
    
    # Extract code from response
    code = response.content
    # Remove markdown code blocks if present
    code = re.sub(r'```python\n(.*?)```', r'\1', code, flags=re.DOTALL)
    code = re.sub(r'```\n(.*?)```', r'\1', code, flags=re.DOTALL)
    
    state["generated_code"] = code
    
    return state

def execute_code(state: ChatDataInsightState) -> ChatDataInsightState:
    """Execute the generated code safely"""
    
    df = state.get("raw_data")
    code = state.get("generated_code", "")
    
    if df is None or not code:
        state["error"] = "No data or code to execute"
        return state
    
    # Create execution environment
    exec_globals = {
        'df': df,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'print': print,
    }
    
    # Capture output
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        # Execute code
        exec(code, exec_globals)
        
        # Capture any printed output
        output = buffer.getvalue()
        state["execution_result"] = output
        
        # Check if a plot was created
        if plt.get_fignums():
            # Save the plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            state["visualization"] = image_base64
            plt.close('all')
        
    except Exception as e:
        state["error"] = f"Code execution error: {str(e)}"
        state["execution_result"] = buffer.getvalue()
    finally:
        sys.stdout = old_stdout
    
    return state

def generate_insights(state: ChatDataInsightState) -> ChatDataInsightState:
    """Use LLM to generate insights from the execution results"""
    
    query_type = state.get("query_type", "analysis")
    execution_result = state.get("execution_result", "")
    has_visualization = state.get("visualization") is not None
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analysis expert. Based on the user's query and the execution results, 
        provide clear, actionable insights. Be concise but thorough. If there's a visualization, 
        describe what it shows. Format your response in a conversational way."""),
        ("human", """User Query: {query}
        
Execution Output: {output}
Visualization Created: {has_viz}

Provide insights based on these results.""")
    ])
    
    response = llm.invoke(prompt.format_messages(
        query=state["current_query"],
        output=execution_result or "No output",
        has_viz="Yes" if has_visualization else "No"
    ))
    
    state["insights"] = response.content
    
    return state

def update_chat_history(state: ChatDataInsightState) -> ChatDataInsightState:
    """Update the chat history with the latest interaction"""
    
    if "messages" not in state:
        state["messages"] = []
    
    # Add user message
    state["messages"].append({
        "role": "user",
        "content": state["current_query"]
    })
    
    # Prepare assistant response
    response_parts = []
    
    if state.get("insights"):
        response_parts.append(state["insights"])
    
    if state.get("execution_result") and state.get("query_type") == "analysis":
        response_parts.append(f"\n\n**Results:**\n```\n{state['execution_result']}\n```")
    
    if state.get("error"):
        response_parts.append(f"\n\n⚠️ Error: {state['error']}")
    
    # Add assistant message
    state["messages"].append({
        "role": "assistant",
        "content": "\n".join(response_parts),
        "visualization": state.get("visualization"),
        "code": state.get("generated_code")
    })
    
    return state

# Routing function
def route_query(state: ChatDataInsightState) -> str:
    """Route based on whether data is loaded"""
    if state.get("raw_data") is None and state.get("file_path"):
        return "load_data"
    elif state.get("error"):
        return "update_chat"
    else:
        return "classify"

# Create the workflow
def create_chat_insights_workflow():
    """Create and compile the LangGraph workflow for chat-based insights"""
    
    # Initialize the graph
    workflow = StateGraph(ChatDataInsightState)
    
    # Add nodes
    workflow.add_node("load_data", load_and_profile_data)
    workflow.add_node("classify", classify_query)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("execute", execute_code)
    workflow.add_node("insights", generate_insights)
    workflow.add_node("update_chat", update_chat_history)
    
    # Define the flow with conditional routing
    workflow.set_conditional_entry_point(route_query)
    
    workflow.add_edge("load_data", "classify")
    workflow.add_edge("classify", "generate_code")
    workflow.add_edge("generate_code", "execute")
    workflow.add_edge("execute", "insights")
    workflow.add_edge("insights", "update_chat")
    workflow.add_edge("update_chat", END)
    
    # Compile the graph
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# Chat interface class
class DataChatbot:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.app = create_chat_insights_workflow()
        self.thread_id = f"chat-{file_path}-{pd.Timestamp.now()}"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        
        # Initialize with file
        initial_state = {
            "file_path": file_path,
            "messages": [],
            "current_query": "Initialize"
        }
        self.app.invoke(initial_state, self.config)
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Process a chat query and return response with visualization if any"""
        
        state = {
            "current_query": query,
            "visualization": None,
            "error": None,
            "generated_code": None,
            "execution_result": None,
            "insights": None
        }
        
        # Run the workflow
        result = self.app.invoke(state, self.config)
        
        # Extract response
        response = {
            "text": result.get("insights", ""),
            "visualization": result.get("visualization"),
            "code": result.get("generated_code"),
            "error": result.get("error")
        }
        
        return response
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get chat history"""
        state = self.app.get_state(self.config)
        return state.values.get("messages", [])

# Example usage
def demo_chatbot():
    """Demonstrate the chatbot with sample data"""
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.normal(1000, 200, 100).cumsum(),
        'customers': np.random.poisson(50, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'product': np.random.choice(['A', 'B', 'C'], 100),
        'satisfaction': np.random.uniform(3, 5, 100)
    })
    df.to_csv('demo_sales_data.csv', index=False)
    
    # Initialize chatbot
    chatbot = DataChatbot('demo_sales_data.csv')
    
    # Example queries
    queries = [
        "What does the data look like? Give me a summary",
        "Create a line plot showing sales over time",
        "Show me the distribution of customers by region with a bar chart",
        "What's the average satisfaction score by product? Create a visualization",
        "Can you show me the correlation between sales and customers with a scatter plot?",
        "Create a box plot comparing satisfaction scores across regions"
    ]
    
    print("=== Data Analysis Chatbot Demo ===\n")
    
    for query in queries:
        print(f"User: {query}")
        response = chatbot.chat(query)
        
        print(f"Assistant: {response['text']}")
        if response['visualization']:
            print("[Visualization created]")
        if response['error']:
            print(f"Error: {response['error']}")
        print("-" * 50)
    
    return chatbot

# Minimal usage example
if __name__ == "__main__":
    # Quick start
    chatbot = DataChatbot("your_data.csv")
    
    # Ask questions
    response = chatbot.chat("Create a bar chart showing top 10 products by sales")
    print(response['text'])
    
    # Get visualization as base64
    if response['visualization']:
        # You can save or display this image
        import base64
        img_data = base64.b64decode(response['visualization'])
        with open('chart.png', 'wb') as f:
            f.write(img_data)
