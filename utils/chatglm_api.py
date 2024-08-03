from typing import Dict, List, Optional

import argparse
import fastapi
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
    
# conv = Conversation(
#     system="A chat between a curious user and an artificial intelligence assistant. ",
#     roles=(["user", "assistant"]),
#     messages=(),
# )


app = fastapi.FastAPI()

headers = {"User-Agent": "FastChat API Server"}



class Request(BaseModel):
    model: str
    prompt: List[str]
    top_p: Optional[float] =0.7
    temperature: Optional[float] = 0.7
    

@app.post("/v1/chat/completions")
def chat_completion(request: Request):
    """Creates a completion for the chat message"""
    payload = generate_payload(request.prompt) 
    content = invoke_example(request.model, payload)
        
    print(content)
    return content
    


from zhipuai import ZhipuAI
 
# your api key
api_key = "699d2724609b25475e617af3e918288d.fj9v0UWEgfcURTm9"
client = ZhipuAI(api_key=api_key)

def invoke_example(model, prompt):
    response = client.chat.completions.create(
	    model=model,
	    messages=prompt,
        top_p=0.7,
        temperature=0.7,
	)
    
    return response.choices[0].message.content


def generate_payload(prompts: List[str]):
    messages = [{"role": "system", "content": prompts[0]}]
    prompts = prompts[1:]
    role = "user"
    for prompt in prompts:
        messages.append({"role": role, "content": prompt})
        role = "assistant" if role == "user" else "user"
    return messages
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGLM-compatible Restful API server.")
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=6000, help="port number")
 
    args = parser.parse_args()
    uvicorn.run("utils.chatglm_api:app", host=args.host, port=args.port, reload=False)
    
    
    
    
    
    