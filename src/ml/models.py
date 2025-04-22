from sqlalchemy import Column, Integer, String
from database import Base

class AbstractInput(Base):
    __tablename__ = "abstract_inputs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    abstract = Column(String)
    keywords = Column(String)
    result = Column(String)