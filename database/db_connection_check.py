import os
import asyncio
from dotenv import load_dotenv
import pymysql
import aiomysql

load_dotenv()

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")

print("\n🔍 Attempting reconnecting the database with configuration:")
print(f"Host: {DB_HOST}")
print(f"Port: {DB_PORT}")
print(f"User: {DB_USERNAME}")
print(f"Database: {TARGET_DB}")

# ---------- 同步 pymysql 测试 ----------
try:
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=TARGET_DB
    )
    print("✅ [pymysql] Successfully connected to database.")
    conn.close()
except Exception as e:
    print("❌ [pymysql] Failed to connect: ", e)


# ---------- 异步 aiomysql 测试 ----------
async def test_aiomysql():
    try:
        conn = await aiomysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            db=TARGET_DB
        )
        print("✅ [aiomysql] Successfully connected to database.")
        conn.close()
    except Exception as e:
        print("❌ [aiomysql] Failed to connect: ", e)


if __name__ == "__main__":
    asyncio.run(test_aiomysql())