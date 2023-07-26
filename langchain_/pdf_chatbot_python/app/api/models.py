from typing import List, Tuple

from pydantic import BaseModel


class Chat(BaseModel):
    question: str
    history: List[Tuple[str, str]]  # [[question, answer]]

