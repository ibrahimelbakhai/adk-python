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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import chromadb
from typing_extensions import override

from google.genai import types

from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

if TYPE_CHECKING:
  from google.adk.sessions.session import Session


_EMBED_DIM = 32


def _embed_text(text: str) -> list[float]:
  """Embeds text into a fixed-size vector using simple hashing."""
  vec = [float(ord(c) % 256) / 255 for c in text[:_EMBED_DIM]]
  if len(vec) < _EMBED_DIM:
    vec.extend([0.0] * (_EMBED_DIM - len(vec)))
  return vec


class ChromaMemoryService(BaseMemoryService):
  """Memory service backed by a persistent Chroma store."""

  def __init__(self, *, persist_directory: str = './data/chroma_store'):
    os.makedirs(persist_directory, exist_ok=True)
    self._client = chromadb.PersistentClient(path=persist_directory)
    self._collection = self._client.get_or_create_collection('session_events')

  @override
  async def add_session_to_memory(self, session: Session):
    docs: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []
    embeddings: list[list[float]] = []
    for idx, event in enumerate(session.events):
      if not event.content or not event.content.parts:
        continue
      text_parts = [p.text for p in event.content.parts if p.text]
      if not text_parts:
        continue
      text = ' '.join(text_parts)
      docs.append(text)
      metadatas.append(
          {
              'app_name': session.app_name,
              'user_id': session.user_id,
              'session_id': session.id,
              'author': event.author,
              'timestamp': _utils.format_timestamp(event.timestamp),
          }
      )
      ids.append(f'{session.id}-{idx}')
      embeddings.append(_embed_text(text))
    if docs:
      self._collection.add(
          ids=ids,
          documents=docs,
          metadatas=metadatas,
          embeddings=embeddings,
      )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    result = self._collection.query(
        query_embeddings=[_embed_text(query)],
        n_results=5,
    )
    response = SearchMemoryResponse()
    documents = result.get('documents', [])
    metadatas = result.get('metadatas', [])
    if documents:
      for doc, metadata in zip(documents[0], metadatas[0]):
        response.memories.append(
            MemoryEntry(
                content=types.Content(
                    role='user', parts=[types.Part.from_text(text=doc)]
                ),
                author=metadata.get('author'),
                timestamp=metadata.get('timestamp'),
            )
        )
    return response
