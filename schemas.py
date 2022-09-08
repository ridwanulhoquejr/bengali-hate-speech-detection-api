from email import message
from tkinter.tix import INTEGER
from pydantic import BaseModel

class Comment(BaseModel):

    message: str

