from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader
from dto.npcDTO import SendDTO, AnswerDTO
from domain.session_manager import session_manager
from domain.npc_chain import conversation_chain, npc_backgrounds
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
name = "users"
session_header = APIKeyHeader(name="X-Session-ID", auto_error=False)

def get_or_create_session(session_id: str = Depends(session_header)):
    if session_id:
        existing_session = session_manager.get_session(session_id)
        if existing_session:
            return session_id  # 기존 세션 ID 반환
    return session_manager.create_session()

@router.post("/send", response_model=AnswerDTO)
async def send(payload: SendDTO, session_id: str = Depends(get_or_create_session)):
    try:
        user_input = payload.content
        npc_id = payload.npc_id

        npc_background = npc_backgrounds.get(npc_id, npc_backgrounds["merchant"])
        conversation_history = session_manager.get_conversation_history(session_id)
        
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Conversation history before: {conversation_history}")
        logger.info(f"User input: {user_input}")

        response = await conversation_chain.generate_response(user_input, npc_background, conversation_history)

        # 대화 기록 업데이트
        session_manager.update_session(session_id, user_input, response.response)
        updated_history = session_manager.get_conversation_history(session_id)
        logger.info(f"Conversation history after: {updated_history}")

        logger.info(f"NPC response: {response.response}")

        return AnswerDTO(content=response.response, session_id=session_id)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")

@router.post("/reset")
async def reset_conversation(session_id: str = Depends(session_header)):
    if session_id:
        session_manager.delete_session(session_id)
    new_session_id = session_manager.create_session()
    return {"message": "Conversation history reset successfully", "new_session_id": new_session_id}

def cleanup_expired_sessions():
    session_manager.cleanup_expired_sessions()