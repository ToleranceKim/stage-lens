from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

# 기본 데이터베이스 클래스 생성
Base = declarative_base()

# 공연 데이터 모델 정의
class Performance(Base):
    __tablename__ = "performances"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    genre = Column(String, nullable=False)