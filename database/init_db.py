import asyncio
from database.connection import engine
from app.models.performance import Base

async def init_db():
    async with engine.begin() as conn:
        # 테이블 생성
        await conn.run_sync(Base.metadata.create_all)

# 비동기 실행
if __name__ == "__main__":
    asyncio.run(init_db())