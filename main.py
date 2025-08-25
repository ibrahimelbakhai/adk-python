# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal FastAPI server for MaestroAgent."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from maestro import MaestroAgent
from maestro.memory.chroma_memory_service import ChromaMemoryService

APP_NAME = "maestro"
USER_ID = "user"

session_service = InMemorySessionService()
memory_service = ChromaMemoryService()
agent = MaestroAgent(name=APP_NAME)
runner = Runner(
    app_name=APP_NAME,
    agent=agent,
    session_service=session_service,
    memory_service=memory_service,
)

app: FastAPI = get_fast_api_app(agents_dir=".", web=False)


class MessageRequest(BaseModel):
  """Request body for sending user messages."""

  message: str


@app.post("/sessions")
async def create_session():
  """Creates a new session and returns its ID."""
  session = await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )
  return {"id": session.id}


@app.post("/sessions/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest):
  """Sends a user message to the agent and returns the response."""
  content = types.Content(
      role="user", parts=[types.Part.from_text(text=request.message)]
  )
  responses: list[str] = []
  async for event in runner.run_async(
      user_id=USER_ID, session_id=session_id, new_message=content
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text
      if text:
        responses.append(text)
  if not responses:
    raise HTTPException(status_code=500, detail="No response from agent")
  return {"response": responses[-1]}


@app.get("/sessions/{session_id}/memory")
async def get_memory(session_id: str, query: str = Query(..., alias="q")):
  """Searches stored memories for the given session."""
  session = await session_service.get_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=session_id
  )
  if not session:
    raise HTTPException(status_code=404, detail="Session not found")
  result = await memory_service.search_memory(
      app_name=session.app_name, user_id=session.user_id, query=query
  )
  memories = []
  for memory in result.memories:
    text = ""
    if memory.content and memory.content.parts:
      part = memory.content.parts[0].text
      if part:
        text = part
    memories.append(
        {"author": memory.author, "text": text, "timestamp": memory.timestamp}
    )
  return {"memories": memories}
