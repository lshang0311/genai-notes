import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt

"""
Ref:
    https://gist.github.com/imartinez/4a826c1f7b7738c9bce8f6d3caba8045
    https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/introducing-llama-2-on-azure/ba-p/3881233 
    
    Tests:
        http://0.0.0.0:8000/?question=<your_question_here>
        http://0.0.0.0:8000/?question=tell me a joke

"""

llms = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Make sure the model path is correct for your system!
    llms["llama"] = LlamaCPP(
        model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf",
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    yield

app = FastAPI(lifespan=lifespan)

def run_llm(question: str) -> AsyncGenerator:
    llm : LlamaCPP = llms["llama"]
    response_iter = llm.stream_complete(question)
    for response in response_iter:
        yield f"data: {response.delta}\n\n"

@app.get("/")
async def root(question: str) -> StreamingResponse:
    print(question)
    return StreamingResponse(run_llm(question), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)