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

import asyncio
from typing import AsyncGenerator
from typing import List

import litellm
from google.genai import types
from typing_extensions import override

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event


class MaestroAgent(BaseAgent):
  """A minimal LLM agent powered by `litellm`."""

  model: str = "gpt-4o"

  def run_prompt(self, prompt: str) -> str:
    """Runs the agent synchronously for a raw prompt.

    Args:
      prompt: The user prompt.

    Returns:
      The model's text response.
    """
    return asyncio.run(self.run_prompt_async(prompt))

  async def run_prompt_async(self, prompt: str) -> str:
    """Runs the agent asynchronously for a raw prompt.

    Args:
      prompt: The user prompt.

    Returns:
      The model's text response.
    """
    response = await litellm.acompletion(
        model=self.model, messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the agent within an ADK invocation context.

    Args:
      ctx: Invocation context supplied by the runner.

    Yields:
      Events containing the model's response.
    """
    messages: List[dict[str, str]] = []
    for event in ctx.session.events:
      if not event.content or not event.content.parts:
        continue
      text = " ".join(p.text for p in event.content.parts if p.text)
      if not text:
        continue
      messages.append({"role": event.author, "content": text})
    if ctx.user_content and ctx.user_content.parts:
      user_text = " ".join(p.text for p in ctx.user_content.parts if p.text)
      if user_text:
        messages.append({"role": "user", "content": user_text})
    # Instead of calling an external LLM, simply echo the user's latest message.
    content_text = ""
    if messages:
      content_text = f"Echo: {messages[-1]['content']}"
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part.from_text(text=content_text)]),
    )

