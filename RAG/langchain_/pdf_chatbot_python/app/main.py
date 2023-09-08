import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import logging

from langchain_.pdf_chatbot_python.app.api.chat_api import chat_router

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.ERROR)

app = FastAPI(openapi_url='/api/v1/chat/openapi.json',
              docs_url='/api/v1/chat/docs',
              debug=True)
background_tasks = BackgroundTasks()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.on_event("startup")
async def startup():
    pass

app.include_router(chat_router, prefix='/api/v1/chat')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
