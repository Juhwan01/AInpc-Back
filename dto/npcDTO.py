from pydantic import BaseModel

class SendDTO(BaseModel):
    npc_id:str
    content: str
    
class AnswerDTO(BaseModel):
    content: str
    session_id: str
