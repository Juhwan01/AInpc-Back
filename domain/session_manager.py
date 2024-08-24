from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, expiry_time: timedelta = timedelta(hours=2)):
        self.sessions: Dict[str, Dict] = {}
        self.expiry_time = expiry_time

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "conversation_history": []
        }
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[str]:
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if datetime.now() - session["last_activity"] < self.expiry_time:
                session["last_activity"] = datetime.now()
                logger.info(f"Retrieved existing session: {session_id}")
                return session_id
            else:
                logger.info(f"Session expired: {session_id}")
                del self.sessions[session_id]
        logger.info(f"Session not found: {session_id}")
        return None

    def update_session(self, session_id: str, user_input: str, npc_response: str):
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session["last_activity"] = datetime.now()
            session["conversation_history"].append({
                "user": user_input,
                "npc": npc_response
            })
            # 대화 기록을 최대 10개로 제한
            session["conversation_history"] = session["conversation_history"][-10:]
            logger.info(f"Updated session: {session_id}")
        else:
            logger.warning(f"Attempted to update non-existent session: {session_id}")

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        if session_id in self.sessions:
            return self.sessions[session_id]["conversation_history"]
        logger.warning(f"Attempted to get history for non-existent session: {session_id}")
        return []

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
        else:
            logger.warning(f"Attempted to delete non-existent session: {session_id}")

    def cleanup_expired_sessions(self):
        now = datetime.now()
        expired_sessions = [sid for sid, session in self.sessions.items() 
                            if now - session["last_activity"] > self.expiry_time]
        for sid in expired_sessions:
            del self.sessions[sid]
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def set_expiry_time(self, new_expiry_time: timedelta):
        self.expiry_time = new_expiry_time
        logger.info(f"Set new expiry time: {new_expiry_time}")

# 전역 SessionManager 인스턴스 생성
session_manager = SessionManager()