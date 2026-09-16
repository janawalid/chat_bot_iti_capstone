from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural-language question")
    history: list[ChatTurn] = Field(
        default_factory=list,
        description=(
            "Recent conversation turns (most recent last), used to resolve "
            "follow-up questions like 'explain how it works' into a "
            "standalone question before retrieval. Optional -- an empty "
            "list behaves exactly as before."
        ),
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]