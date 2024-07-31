from typing import Union, Dict, List, Any,Optional

import argparse
import json
import logging

import fastapi
import httpx
import uvicorn


from pydantic import BaseModel

class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    
    def __init__(self,system, roles, messages):
        self.system = system
        self.roles = roles
        self.messages = messages
    
conv = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. ",
    roles=(["user", "assistant"]),
    messages=(),
)


app = fastapi.FastAPI()

headers = {"User-Agent": "FastChat API Server"}



class Request(BaseModel):
    model: str
    prompt: List[Dict[str, str]]
    top_p: Optional[float] =0.7
    temperature: Optional[float] = 0.7
    

@app.post("/v1/chat/completions")
def chat_completion(request: Request):
    """Creates a completion for the chat message"""
    conv.messages =[]
    payload = generate_payload(request.prompt) 
    content = invoke_example(request.model, payload)

    generate_payload(content)
        
        
    print('a',content)
    return content[0]['content']
    


from zhipuai import ZhipuAI
 
# your api key
api_key = "699d2724609b25475e617af3e918288d.fj9v0UWEgfcURTm9"
client = ZhipuAI(api_key=api_key)

def invoke_example(model,prompt):
    response = client.chat.completions.create(
	    model="GLM-4",
	    messages=prompt,
        top_p=0.7,
        temperature=0.7,
	)
    
    return [dict(response.choices[0].message)]



def generate_payload(messages: List[Dict[str, str]]):
    conv.messages = []
    for message in messages:
       
        msg_role = message["role"]
        
        if msg_role == "user":
            conv.messages.append({'role': conv.roles[0], 'content': message["content"]})
        elif msg_role == "assistant":
            conv.messages.append({'role': conv.roles[1], 'content': message["content"]})
        else:
            raise ValueError(f"Unknown role: {msg_role}")
    
    return conv.messages
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGLM-compatible Restful API server.")
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=6000, help="port number")
 
    args = parser.parse_args()
    uvicorn.run("utils.chatglm_api:app", host=args.host, port=args.port, reload=False)
    
    
    
    
    
    