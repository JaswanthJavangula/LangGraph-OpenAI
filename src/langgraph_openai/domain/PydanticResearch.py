from pydantic import BaseModel

class ResearchPydanticDomain(BaseModel):
    topic_title: str
    summary: str
    sources: list[str]
    dateTime: str
