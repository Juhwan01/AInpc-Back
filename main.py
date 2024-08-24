import logging
import traceback
from urllib.request import Request
import uvicorn
from langchain_teddynote import logging as log

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from routers import router as main_router
from domain.npc_chain import conversation_chain

# 프로젝트 이름을 입력합니다.
log.langsmith("원하는 프로젝트명")

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router=main_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

async def add_cors_to_response(
    request: Request, response: JSONResponse
) -> JSONResponse:
    origin = request.headers.get("origin")

    if origin:
        cors = CORSMiddleware(
            app=app,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        response.headers.update(cors.simple_headers)
        has_cookie = "cookie" in request.headers

        if cors.allow_all_origins and has_cookie:
            response.headers["Access-Control-Allow-Origin"] = origin
        elif not cors.allow_all_origins and cors.is_allowed_origin(
            origin=origin,
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers.add_vary_header("Origin")
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logging.error(traceback.format_exc())
    response = JSONResponse(status_code=500, content={"context": str(exc)})
    return await add_cors_to_response(request=request, response=response)

@app.on_event("startup")
async def startup_event():
    # 앱 시작 시 conversation_chain이 이미 초기화되어 있으므로 별도의 작업이 필요 없습니다.
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)