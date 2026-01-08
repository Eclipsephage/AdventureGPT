"""
Types for LLM messaging used across AdventureGPT.

This file defines a minimal, provider-agnostic message format so agent logic
does not depend on a specific SDK.
"""

from __future__ import annotations

from typing import List, Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """
    Provider-agnostic chat message.
    """

    role: Role
    content: str


Messages = List[Message]

