from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from decouple import config

# 환경 변수에서 데이터베이스 URL 가져오기
DATABASE_URL = config("DATABASE_URL")

print(f"DATABASE_URL: {DATABASE_URL}")

# SQLAlchemy 비동기 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True)

# 세션 생성기
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 세션을 제공하는 함수
async def get_db():
    async with async_session() as session:
        yield session