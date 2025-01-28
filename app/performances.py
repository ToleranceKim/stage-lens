from fastapi import APIRouter

# 라우터 생성
router = APIRouter(
    prefix="/performances",
    tags=["performances"]
)

# 샘플 공연 데이터
performances = [
    {"id": 1, "title": "Phantom of the Opera", "genre": "Musical"},
    {"id": 2, "title": "Les Misérables", "genre": "Drama"},
]

@router.get("/")
def get_all_performances():
    """모든 공연 데이터 반환"""
    return {"performances": performances}

@router.get("/{performance_id}")
def get_performance(performance_id: int):
    """특정 공연 데이터를 ID로 조회"""
    for performance in performances:
        if performance["id"] == performance_id:
            return performance
    return {"error": "Performance not found"}
