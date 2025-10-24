# main.py
from dotenv import load_dotenv
load_dotenv()

import sys
import uuid
import asyncio
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from chatbot import Chatbot

# -----------------------------
# Windows Asyncio Fix
# -----------------------------
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -----------------------------
# FastAPI app & CORS
# -----------------------------
app = FastAPI(
    title="Session-Based RAG Chatbot API",
    description="Session-based RAG Chatbot API with WebSocket support",
    version="1.1.0"
)

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Session storage
# -----------------------------
chatbot_sessions = {}  # {session_id: Chatbot instance or None if failed}

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Session-Based RAG Chatbot API!", "status": "Ready"}

@app.get("/create_session")
def create_session():
    return {"session":str(uuid.uuid4())}


@app.get("/session_status/{session_id}")
def session_status(session_id: str):
    """
    Returns the current status of a chatbot session.
    Status can be:
      - initializing (session exists but chatbot not ready)
      - ready (chatbot instance ready)
      - failed (chatbot initialization failed)
    """
    if session_id not in chatbot_sessions:
        return {"status": "not_found"}
    
    chatbot = chatbot_sessions[session_id]
    if chatbot is None:
        return {"status": "initializing"}
    elif chatbot == "err":
        chatbot = None
        return {"status": "failed"}
    
    return {"status": "ready"}



# -----------------------------
# Helper: Run async init in background
# -----------------------------
def run_chatbot_init(session_id, urls, llm_model, embedding_model, api_key):
    asyncio.create_task(initialize_chatbot(session_id, urls, llm_model, embedding_model, api_key))

# -----------------------------
# Scrape & initialize chatbot
# -----------------------------
@app.post("/scrape/")
async def scrape_and_load(response: dict, background_tasks: BackgroundTasks):
    session_id = response.get("session_id")
    urls = response.get("urls")
    llm_model = response.get("llm_model", "TheBloke/Llama-2-7B-Chat-GGML")
    embedding_model = response.get("embedding_model", "BAAI/bge-small-en")
    api_key = response.get("api_key", None)

    if not urls:
        raise HTTPException(status_code=400, detail="urls are required.")

    if session_id in chatbot_sessions:
        return {"message": f"Chatbot for session {session_id} already initialized.", "session_id": session_id}

    # Mark session as initializing
    chatbot_sessions[session_id] = None

    # Use a **blocking wrapper** to run async in thread safely
    async def init_wrapper():
        try:
            await initialize_chatbot(session_id, urls, llm_model, embedding_model, api_key)
        except Exception as e:
            logging.error(f"[{session_id}] Initialization error: {e}", exc_info=True)
            chatbot_sessions[session_id] = None

    background_tasks.add_task(init_wrapper)

    logging.info(f"[{session_id}] Chatbot initialization scheduled in background.")
    return {"message": "Chatbot initialization started.", "session_id": session_id}

# -----------------------------
# Initialize chatbot
# -----------------------------
async def initialize_chatbot(session_id, urls, llm_model, embedding_model, api_key):
    try:
        logging.info(f"[{session_id}] Initializing chatbot...")
        chatbot = Chatbot(
            url=urls,
            llm_model=llm_model,
            embedding_model=embedding_model,
            api_key=api_key
        )
        await chatbot.initialize()

        chatbot_sessions[session_id] = chatbot
        logging.info(f"[{session_id}] Chatbot ready.")
    except NotImplementedError as e:
        logging.error(f"[{session_id}] Playwright async not supported on Windows: {e}", exc_info=True)
        chatbot_sessions[session_id] = None
    except Exception as e:
        logging.error(f"[{session_id}] Initialization failed: {e}", exc_info=True)
        chatbot_sessions[session_id] = "err"

# -----------------------------
# WebSocket endpoint
# -----------------------------
@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logging.info(f"[{session_id}] WebSocket connected.")

    try:
        # Wait until chatbot is ready
        while session_id not in chatbot_sessions or chatbot_sessions[session_id] is None:
            await websocket.send_json({"text": "Initializing chatbot, please wait..."})
            await asyncio.sleep(1)

        chatbot_instance = chatbot_sessions[session_id]
        if chatbot_instance is None:
            await websocket.send_json({
                "text": "Chatbot initialization failed. Likely due to Playwright async issue on Windows."
            })
            return

        await websocket.send_json({"text": f"Chatbot session {session_id} is ready! You can start chatting."})

        while True:
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                continue

            response_text = await chatbot_instance.query(query)
            await websocket.send_json({"text": response_text})

    except WebSocketDisconnect:
        logging.info(f"[{session_id}] WebSocket disconnected.")
    except Exception as e:
        logging.error(f"[{session_id}] WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"text": "An unexpected server error occurred."})
        except:
            pass

# -----------------------------
# Run with: uvicorn main:app --reload
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
