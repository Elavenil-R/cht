from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

Message = dict  # {"role": "user"|"assistant", "content": "..."}

@dataclass
class SlidingWindowMemory:
    max_turns: int = 10
    _store: Dict[str, List[Message]] = field(default_factory=dict)

    @property
    def max_messages(self) -> int:
        return self.max_turns * 2  # 10 turns = 20 messages

    def get(self, session_id: str) -> List[Message]:
        return self._store.get(session_id, [])

    def add_user(self, session_id: str, text: str) -> None:
        self._append(session_id, {"role": "user", "content": text})

    def add_assistant(self, session_id: str, text: str) -> None:
        self._append(session_id, {"role": "assistant", "content": text})

    def _append(self, session_id: str, msg: Message) -> None:
        hist = self._store.setdefault(session_id, [])
        hist.append(msg)
        # keep only last N messages
        if len(hist) > self.max_messages:
            del hist[:-self.max_messages]