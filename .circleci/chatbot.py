https://github.com/vllm-project/vllm/blob/main/examples/online_serving/api_client.py
https://docs.vllm.ai/en/latest/examples/online_serving/api_client.html

import json
import os
import requests
import datetime

filename = "config.json"

def save_properties_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_properties_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Request a new token if the existing token has expired
def getAuthToken(clientId = "", clientSecret = "", address = "") -> str:
    json_object = load_properties_json(filename)
    # print(json_object)

    if len(json_object) > 0:
        if datetime.datetime.now() < (datetime.datetime.strptime(json_object["access_token_created_at"], "%Y-%m-%d %H:%M:%S")) + datetime.timedelta(seconds=json_object["expires_in"]):
            # return current token
            # print("Returning existing token " + json_object["access_token"])
            return json_object["access_token"]

    payload = json.dumps(
        {
            "client_id": clientId,
            "client_secret": clientSecret,
            "grant_type": "client_credentials"
        }
    )

    headers = {
        "Content-Type": "application/json",
    }

    response = sendHttpRequest(data=payload,
                               headers=headers,
                               method="POST",
                               address=address,
                               endpoint="/v2/oauth2/token")

    # print(response)

    json_data = response.text
    json_object = json.loads(json_data)

    if response.status_code != 200:
        print("Auth failed!")
        return ""

    # print("Saving new token")
    
    json_object["access_token_created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_properties_json(filename, json_object)
        
    return json_object["access_token"]

def sendHttpRequest(
                    data,
                    headers: dict[str, str],
                    method: str,    
                    address: str,
                    endpoint: str,
                    files: list[tuple[str, tuple[str, bytes, str]]] = None,
                    params: dict[str, str] = None,
                    stream: bool = False) -> bytearray:

    protocol = "http"
    if os.getenv('HORIZON_API_SECURE', 'false').lower() == 'true':
        protocol = "https"

    url = f"{protocol}://{address}{endpoint}"
    # print("Url: ", url)
    
    if endpoint == '/v2/document/online/completions':
        response = requests.request(
            # Set verify to False to ignore the SSL cert, otherwise provide a path (here, an env var)
            # method=method, url=url, headers=header, data=data, files=files, params=params, stream=stream, verify=False
            method=method, url=url, headers=headers, data=data, files=files, params={"qos":"cheap"}, stream=stream, verify="cacert.crt"
        )
    elif endpoint == '/v2/document/chats':
        response = requests.request(
            # Set verify to False to ignore the SSL cert, otherwise provide a path (here, an env var)
            # method=method, url=url, headers=header, data=data, files=files, params=params, stream=stream, verify=False
            method=method, url=url, headers=headers, data=data, params=params, stream=stream, verify="cacert.crt"
        )
    
    elif endpoint == '/v2/text/chats':
        
        # data = {"messages":data}
        # print("data to text/chats: \n",data)
        response = requests.request(
                # Set verify to False to ignore the SSL cert, otherwise provide a path (here, an env var)
                # method=method, url=url, headers=header, data=data, files=files, params=params, stream=stream, verify=False
                method=method, url=url, headers=headers, json=data, params=params, stream=stream, verify="cacert.crt"
        )
        # print(response.content)
    else:
        response = requests.request(
            # Set verify to False to ignore the SSL cert, otherwise provide a path (here, an env var)
            # method=method, url=url, headers=header, data=data, files=files, params=params, stream=stream, verify=False
            method=method, url=url, headers=headers, data=data, params=params, stream=stream, verify="cacert.crt"
        )
    # print("response from SendRequest", response.content)

    if response.status_code != 200:
        print(f"HTTP request failed. Endpoint {endpoint} Error: {response.status_code}\n")
        # print("Response in error",response.content)
        return bytearray()
    else:
        # print(f"Returning {response}")
        return response
