import importlib
from pathlib import Path

from fastapi import APIRouter

router_files = Path(__file__).parent.rglob("*.py")

routers = [
    importlib.import_module(f"routers.{file.parent.name}.{file.stem}")
    for file in router_files
    if file.stem != "__init__" and file.parent.name != "base"
]

# 하위 라우터를 모으기 위한 중앙 라우터
router = APIRouter()

for r in routers:
    router.include_router(
        r.router,
        prefix="/api",
        tags=[
            f"{r.name}",
        ],
        dependencies=[],
    )
