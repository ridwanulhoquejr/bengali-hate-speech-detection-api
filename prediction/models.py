import array
from email import message
from sqlalchemy import Column, Integer, String, ForeignKey
from .database import Base



class Prediction(Base):

    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String)
    label = Column(Integer)



 


