from fastapi import APIRouter
from pydantic import BaseModel

# APIRouter 생성
router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"],
)

# 추천 요청 데이터 모델
class RecommendationRequest(BaseModel):
    story:str

@router.post("/")
def recommend_performances(request: RecommendationRequest):
    """
    줄거리에 기반한 추천 공연 반환
    """
    # 간단한 샘플 추천
    recommendations = [
        {"id": 1, "title": "Phantom of the Opera", "genre": "Musical"},
        {"id": 3, "title": "Hamilton", "genre": "Musical"},
    ]
    return {"recommendations": recommendations}