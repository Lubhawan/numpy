# Define a custom LLM
import json
import os
from pyexpat.errors import messages
from sunau import Au_read
from this import d
from ai.chatbot.horizon_dev.utils import getAuthToken, sendHttpRequest
from dotenv import load_dotenv

# from ai.chatbot.prompts.prompts import ge

load_dotenv()

class TextChatCompletionsLLM:

    def _call(self, payload=dict(), files=None, params="", endpoint="", content_type=None, stream=False, **kwargs) -> str:
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

    
    def _llm_type(self) -> str:
        return "TextChatCompletionsLLM"


https://tablegpt.github.io/tablegpt-agent/howto/normalize-datasets/
Normalize Datasets - TableGPT Agent
 
