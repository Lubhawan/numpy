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

class HorizonChatLLM(BaseChatModel):
    """
    A LangChain-compatible chat model that uses Horizon API with simple message format.
    """
    
    model_name: str = "horizon-chat"
    endpoint: str = ""  # Your endpoint
    streaming: bool = False
    
    class Config:
        """Configuration for this model."""
        extra = "allow"
    
    @property
    def _llm_type(self) -> str:
        """Return type of language model."""
        return "horizon_chat"
    
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
    
    def _create_payload(self, messages: List[BaseMessage], stream: bool = False) -> Dict[str, Any]:
        """Create the payload for Horizon API."""
        # Convert messages to the format expected by your API
        formatted_messages = [self._convert_message_to_dict(msg) for msg in messages]
        
        # Create payload with only messages and stream parameters
        payload = {
            "messages": formatted_messages,
            "stream": str(stream)  # Convert boolean to string as per your API requirement
        }
                
        return payload
    
    def _call_api(self, payload: Dict[str, Any]) -> str:
        """Make the actual API call to Horizon."""
        if not os.getenv('HORIZON_CLIENT_ID'):
            raise ValueError("HORIZON_CLIENT_ID is not set in the environment.")
        if not os.getenv('HORIZON_CLIENT_SECRET'):
            raise ValueError("HORIZON_CLIENT_SECRET is not set in the environment.")
        if not os.getenv('HORIZON_GATEWAY'):
            raise ValueError("HORIZON_GATEWAY is not set in the environment.")
        
        authToken = getAuthToken(
            os.getenv('HORIZON_CLIENT_ID', ''),
            os.getenv('HORIZON_CLIENT_SECRET', ''),
            os.getenv('HORIZON_GATEWAY', '')
        )
        
        headers = {
            "Authorization": f"Bearer {authToken}"
        }
        
        # Determine if streaming based on payload
        stream = payload.get("stream", "False") == "True"
        
        try:
            response = sendHttpRequest(
                data=payload,
                files=None,
                params="",
                headers=headers,
                method="POST",
                address=os.getenv('HORIZON_GATEWAY', ''),
                endpoint=self.endpoint,
                stream=stream
            )
            
            response_data = json.loads(response.text)
            
            # Extract the message content from the response
            # Based on your original code, it seems the response has a "message" field
            return response_data.get("message", "")
                
        except Exception as e:
            print(f"Error in API call: {e}")
            raise
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from the Horizon API."""
        # Create payload with stream=False for regular generation
        payload = self._create_payload(messages, stream=False)
        
        response_content = self._call_api(payload)
        
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
        """Stream responses from the Horizon API."""
        # Create payload with stream=True for streaming
        payload = self._create_payload(messages, stream=True)
        
        # For now, this returns the full response
        # You'll need to implement actual streaming logic based on how your API returns streamed data
        response_content = self._call_api(payload)
        
        message = AIMessage(content=response_content)
        yield ChatGeneration(message=message)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "endpoint": self.endpoint,
        }
