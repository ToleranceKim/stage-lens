from decouple import config

# DATABASE_URL 읽기
try:
    DATABASE_URL = config("DATABASE_URL")
    print(f"DATABASE_URL from .env: {DATABASE_URL}")
except Exception as e:
    print(f"Error loading DATABASE_URL: {e}")