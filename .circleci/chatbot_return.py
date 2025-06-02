import json
import os
from typing import Any, List, Optional, Iterator, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from ai.chatbot.horizon_dev.utils import getAuthToken, sendHttpRequest
from dotenv import load_dotenv

load_dotenv()

class TextChatCompletionsLLM(BaseChatModel):
    """
    A LangChain-compatible chat model that uses Horizon API.
    Maintains original class name and method signatures.
    """
    
    model_name: str = "horizon-chat"
    endpoint: str = ""  # Your endpoint
    content_type: Optional[str] = None
    params: str = ""
    
    class Config:
        """Configuration for this model."""
        extra = "allow"
    
    def __call__(self, payload=dict(), files=None, params="", endpoint="", content_type=None, stream=False, **kwargs) -> str:
        """Original __call__ method signature maintained for backward compatibility."""
        if not os.getenv('HORIZON_CLIENT_ID'):
            raise ValueError("HORIZON_CLIENT_ID is not set in the environment.")
        if not os.getenv('HORIZON_CLIENT_SECRET'):
            raise ValueError("HORIZON_CLIENT_SECRET is not set in the environment.")
        if not os.getenv('HORIZON_GATEWAY'):
            raise ValueError("HORIZON_GATEWAY is not set in the environment.")
        
        authToken = getAuthToken(os.getenv('HORIZON_CLIENT_ID', ''),
                                 os.getenv('HORIZON_CLIENT_SECRET', ''),
                                 os.getenv('HORIZON_GATEWAY', ''))
        
        try:
            response = self.api_call(authToken=authToken,
                                     payload=payload,
                                     files=files,
                                     params=params,
                                     endpoint=endpoint,
                                     stream=stream)
        except Exception as e:
            print("Error in API call: ", e)
            return None
        return response
    
    def api_call(self, authToken: str,
                 payload: dict[str, list[dict]],
                 files: list[tuple[str, tuple[str, bytes, str]]],
                 params: dict[str, str],
                 endpoint: str,
                 stream: bool=False) -> str:
        """Original api_call method maintained."""
        headers = {
            "Authorization": "Bearer " + authToken
        }
        # Ensure files is a list of tuples
        if files and not isinstance(files, list):
            raise ValueError("The 'files' parameter must be a list of tuples.")
        
        try:
            response = sendHttpRequest(data=payload, 
                                       files=files, 
                                       params=params,
                                       headers=headers,
                                       method="POST",
                                       address=os.getenv('HORIZON_GATEWAY', ''),
                                       endpoint=endpoint,
                                       stream=stream)
            response_data = json.loads(response.text)
            return response_data.get("message", "")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing API response: {e}")
            return ""
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model - using your original method name."""
        return "TextChatCompletionsLLM"
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, str]:
        """Convert a LangChain message to Horizon API format."""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        else:
            # Default to user role for other message types
            return {"role": "user", "content": str(message.content)}
    
    def _create_payload_from_messages(self, messages: List[BaseMessage], stream: bool = False) -> Dict[str, Any]:
        """Create the payload for Horizon API from LangChain messages."""
        # Convert messages to the format expected by your API
        formatted_messages = [self._convert_message_to_dict(msg) for msg in messages]
        
        # Create payload with only messages and stream parameters
        payload = {
            "messages": formatted_messages,
            "stream": str(stream)  # Convert boolean to string as per your API requirement
        }
                
        return payload
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the Horizon API - required by LangChain."""
        # Create payload with stream=False for regular generation
        payload = self._create_payload_from_messages(messages, stream=False)
        
        # Use the original __call__ method
        response_content = self.__call__(
            payload=payload,
            files=None,
            params=self.params,
            endpoint=self.endpoint,
            content_type=self.content_type,
            stream=False
        )
        
        message = AIMessage(content=response_content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream responses from the Horizon API - required by LangChain."""
        # Create payload with stream=True for streaming
        payload = self._create_payload_from_messages(messages, stream=True)
        
        # Use the original __call__ method with streaming
        response_content = self.__call__(
            payload=payload,
            files=None,
            params=self.params,
            endpoint=self.endpoint,
            content_type=self.content_type,
            stream=True
        )
        
        message = AIMessage(content=response_content)
        yield ChatGeneration(message=message)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "endpoint": self.endpoint,
        }


# Alternative: If you prefer to keep your original class completely unchanged
# and create a wrapper instead
class LangChainTextChatCompletionsLLM(BaseChatModel):
    """
    A wrapper that makes TextChatCompletionsLLM compatible with LangChain.
    """
    
    base_llm: TextChatCompletionsLLM
    endpoint: str = ""
    params: str = ""
    content_type: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the original TextChatCompletionsLLM
        self.base_llm = TextChatCompletionsLLM()
    
    @property
    def _llm_type(self) -> str:
        return "TextChatCompletionsLLM"
    
    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, str]:
        """Convert a LangChain message to Horizon API format."""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        else:
            return {"role": "user", "content": str(message.content)}
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the original TextChatCompletionsLLM."""
        # Convert messages
        formatted_messages = [self._convert_message_to_dict(msg) for msg in messages]
        
        payload = {
            "messages": formatted_messages,
            "stream": "False"
        }
        
        # Call the original class
        response_content = self.base_llm(
            payload=payload,
            files=None,
            params=self.params,
            endpoint=self.endpoint,
            content_type=self.content_type,
            stream=False
        )
        
        message = AIMessage(content=response_content or "")
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream responses."""
        formatted_messages = [self._convert_message_to_dict(msg) for msg in messages]
        
        payload = {
            "messages": formatted_messages,
            "stream": "True"
        }
        
        response_content = self.base_llm(
            payload=payload,
            files=None,
            params=self.params,
            endpoint=self.endpoint,
            content_type=self.content_type,
            stream=True
        )
        
        message = AIMessage(content=response_content or "")
        yield ChatGeneration(message=message)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "params": self.params,
        }


# Usage example with TableGPT
if __name__ == "__main__":
    from pathlib import Path
    from pybox import AsyncLocalPyBoxManager
    from tablegpt.agent import create_tablegpt_graph
    from tablegpt import DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR
    
    # Option 1: Use the modified TextChatCompletionsLLM directly
    llm = TextChatCompletionsLLM(
        model_name="TableGPT2-7B",
        endpoint="/your/chat/endpoint",  # Set your actual endpoint here
    )
    
    # Option 2: Use the wrapper if you want to keep original class unchanged
    # llm = LangChainTextChatCompletionsLLM(
    #     endpoint="/your/chat/endpoint",
    # )
    
    # For normalize_llm
    normalize_llm = TextChatCompletionsLLM(
        model_name="YOUR_NORMALIZE_MODEL_NAME",
        endpoint="/your/chat/endpoint",
    )
    
    pybox_manager = AsyncLocalPyBoxManager(
        profile_dir=DEFAULT_TABLEGPT_IPYKERNEL_PROFILE_DIR
    )
    
    agent = create_tablegpt_graph(
        llm=llm,
        pybox_manager=pybox_manager,
        normalize_llm=normalize_llm,
        session_id="some-session-id",
    )
    
    # Test the LLM directly
    from langchain_core.messages import HumanMessage
    
    # Test basic chat
    messages = [HumanMessage(content="Hello, how are you?")]
    response = llm.invoke(messages)
    print(response.content)
    
    # You can also still use your original method directly
    # response = llm(payload={"messages": [{"role": "user", "content": "Hello"}], "stream": "False"})
    # print(response)
