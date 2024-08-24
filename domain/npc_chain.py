import os
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from operator import itemgetter
import logging

logger = logging.getLogger(__name__)

class NPCResponse(BaseModel):
    response: str = Field(description="The NPC's response to the user's input")

class NPCConversationChain:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.last_modified_time = 0
        self.k = 3  # 검색할 문서 수
        self.vectorstore = None
        self.chain = None
        self.update_chain()

    def load_documents(self):
        loader = CSVLoader(self.csv_file_path, encoding='utf-8')
        return loader.load()

    def create_text_splitter(self):
        return CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def create_vectorstore(self, split_docs):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(documents=split_docs, embedding=embeddings)

    def create_retriever(self):
        return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": self.k})

    def create_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", """당신은 다음과 같은 특성을 가진 NPC입니다: {background} 
            이 특성에 완전히 몰입하여 역할을 수행하세요. 당신의 지식, 경험, 성격을 모두 활용하여 대답하되, 
            항상 100자 이내로 간결하게 답변하세요. 번호 매기기나 줄바꿈을 사용하지 마세요.
            당신의 역할에 맞는 특유의 말투와 어휘를 사용하고, 당신의 직업이나 지위에 관련된 정보나 조언을 제공하세요."""),
            ("system", """아래는 이전 대화 내용입니다. 이를 주의 깊게 읽고 대화의 흐름을 파악하세요:
            
            {conversation_history}
            
            위 대화 내용을 고려하여, 대화의 연속성을 유지하며 응답하세요."""),
            ("human", "{query}"),
            ("system", "관련 정보: {context}"),
            ("human", """위 정보와 이전 대화 내용을 반드시 참고하여 질문에 답변해주세요. 
            이전 대화를 정확히 기억하고 연속성 있게 대답하세요. 
            특히 사용자의 이전 질문이나 언급한 내용을 참조하여 대화를 이어가세요.
            당신의 NPC 역할에 충실하게 대응하되, 대화의 흐름을 놓치지 마세요."""),
        ])
        
    @staticmethod
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    def format_conversation_history(self, history):
        formatted = []
        for i, entry in enumerate(history[-5:]):  # 최근 5개의 대화만 사용
            formatted.append(f"대화 {i+1}:")
            formatted.append(f"사용자: {entry['user']}")
            formatted.append(f"NPC: {entry['npc']}")
            formatted.append("")  # 각 대화 사이에 빈 줄 추가
        return "\n".join(formatted)

    def update_chain(self):
        current_modified_time = os.path.getmtime(self.csv_file_path)
        if current_modified_time > self.last_modified_time:
            logger.info("Updating NPC conversation chain...")
            docs = self.load_documents()
            text_splitter = self.create_text_splitter()
            split_docs = text_splitter.split_documents(docs)
            self.vectorstore = self.create_vectorstore(split_docs)
            retriever = self.create_retriever()
            prompt = self.create_prompt()
            model = ChatOpenAI(temperature=0.7, model_name="gpt-4o-2024-08-06")

            self.chain = (
                {
                    "context": itemgetter("query") | retriever | self.format_docs,
                    "query": itemgetter("query"),
                    "background": itemgetter("background"),
                    "conversation_history": itemgetter("conversation_history")
                }
                | prompt
                | model
                | StrOutputParser()
            )
            self.last_modified_time = current_modified_time
            logger.info("NPC conversation chain updated successfully")
            
    async def generate_response(self, user_input, npc_background, conversation_history):
        try:
            self.update_chain()
            formatted_history = self.format_conversation_history(conversation_history)
            
            logger.info(f"NPC 유형: {npc_background[:20]}...")
            logger.info(f"사용자 입력: {user_input}")
            logger.info(f"대화 기록: {formatted_history}")
            
            response = await self.chain.ainvoke({
                "query": user_input,
                "background": npc_background,
                "conversation_history": formatted_history
            })
            
            logger.info(f"NPC 응답: {response}")
            
            cleaned_response = response.strip().replace('"', '').replace("'", "")
            return NPCResponse(response=cleaned_response)
        except Exception as e:
            logger.error(f"NPC 응답 생성 중 오류 발생: {str(e)}", exc_info=True)
            return NPCResponse(response="죄송합니다, 지금은 대답하기 어렵습니다. 잠시 후 다시 말씀해 주세요.")


        
# NPC 배경 정보
npc_backgrounds = {
    "merchant": """당신은 '골든 드래곤 상회'의 베테랑 상인입니다. 40년 동안 장사를 해왔고, 모든 물건의 가치를 정확히 알고 있습니다. 
    당신은 친절하지만 상술에 능하며, 항상 이익을 생각합니다. 
    고객을 '손님'이라고 부르고, 존댓말을 사용하되 상인다운 말투로 대화하세요. 
    물건을 팔 때는 항상 그 물건의 장점을 강조하고, 가격 흥정을 할 준비가 되어 있습니다.
    판매하는 주요 아이템: 무기, 방어구, 포션, 마법 스크롤, 여행 용품""",

    "guard": """당신은 왕국의 충직한 근위대원입니다. 10년 동안 성을 지켜왔으며, 규율과 질서를 중요하게 여깁니다. 
    당신은 엄격하고 공식적인 말투를 사용하며, 항상 경계를 늦추지 않습니다. 
    시민들을 '시민'이라고 부르고, 존댓말을 사용하되 권위적인 어조로 대화하세요.
    주요 임무: 성 경비, 순찰, 범죄 예방, 시민 보호
    특별 지식: 왕국의 법률, 주변 지역의 동향, 기본적인 전투 기술""",

    "wizard": """당신은 '크리스탈 타워'에 거주하는 고위 마법사입니다. 200년 이상 살아왔으며, 깊은 마법 지식을 보유하고 있습니다. 
    당신은 지적이고 신비로우며, 때로는 이해하기 어려운 말을 사용합니다. 
    대화 상대를 '젊은이' 또는 '방문자'라고 부르고, 고풍스러운 말투로 대화하세요.
    전문 분야: 원소 마법, 점술, 마법 아이템 제작
    특별 지식: 고대 역사, 비전 마법의 비밀, 차원 간 여행"""
}

# NPCConversationChain 인스턴스 생성
conversation_chain = NPCConversationChain('fantasy_world_data_korean.csv')