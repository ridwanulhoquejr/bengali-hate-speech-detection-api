from pydantic import BaseModel


class Comment(BaseModel):

    message: str