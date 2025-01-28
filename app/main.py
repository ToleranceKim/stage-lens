from fastapi import FastAPI
from app.performances import router as performances_router
from app.recommendations import router as recommendations_router

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Stage Lens API",
    descriptions = "API for performance recommendations",
    version="1.0.0",
)

# 라우터 등록
app.include_router(performances_router) # 공연 데이터 라우터
app.include_router(recommendations_router) # 추천 시스템 라우터

# 기본 엔드포인트
@app.get("/")
def read_root():
    return {"message": "Welcome to Stage-Lens!"}


