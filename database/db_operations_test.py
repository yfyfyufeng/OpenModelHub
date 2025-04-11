import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from database import Base, clear_all_tables
from database_interface import *

# ========= Load Env =========
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")

DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"

# ========= Create Tables with Async Engine =========
async def create_async_db_and_tables(async_engine):
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("✅ 使用 async_engine 初始化数据库结构完成")

# ========= Reset DB =========
async def reset_database():
    async_engine = create_async_engine(DATABASE_URL, echo=True)
    await create_async_db_and_tables(async_engine)
    async_session = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        await clear_all_tables(lambda: session)

# ========= Run All Tests =========
async def run_tests(session: AsyncSession):
    # -----------------------------
    # Affiliation
    # -----------------------------
    affil = await create_affiliation(session, "OpenAI")
    assert affil.affil_name == "OpenAI"
    affil = await update_affiliation(session, affil.affil_id, {"affil_name": "OpenAI Research"})
    assert affil.affil_name == "OpenAI Research"
    assert await delete_affiliation(session, affil.affil_id) is True

    # -----------------------------
    # User
    # -----------------------------
    user = await create_user(session, {"user_name": "Alice", "affiliate": "TestLab"})
    user_id = user.user_id
    user = await update_user(session, user_id, {"user_name": "Alice A."})
    assert user.user_name == "Alice A."
    users = await list_users(session)
    assert any(u.user_id == user_id for u in users)

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = await create_dataset(session, {
        "ds_name": "COCO",
        "ds_size": 50000,
        "media": "image",
        "task": 1
    })
    dataset_id = dataset.ds_id
    dataset = await update_dataset(session, dataset_id, {"ds_size": 55000})
    assert dataset.ds_size == 55000
    datasets = await list_datasets(session)
    assert any(ds.ds_id == dataset_id for ds in datasets)

    # -----------------------------
    # Model
    # -----------------------------
    model = await create_model(session, {
        "model_name": "YOLOv7",
        "param_num": 64000000,
        "media_type": "image",
        "arch_name": "CNN"
    })
    model_id = model.model_id
    model = await update_model(session, model_id, {"model_name": "YOLOv7x"})
    assert model.model_name == "YOLOv7x"

    # -----------------------------
    # Linking
    # -----------------------------
    await link_model_author(session, model_id, user_id)
    await link_model_dataset(session, model_id, dataset_id)
    await add_task_to_model(session, model_id, "Detection")
    await link_user_dataset(session, user_id, dataset_id)
    affil = await create_affiliation(session, "OpenAI")
    await link_user_affiliation(session, user_id, affil.affil_id)

    # -----------------------------
    # Deletion
    # -----------------------------
    assert await delete_model(session, model_id) is True
    assert await delete_user(session, user_id) is True
    assert await delete_dataset(session, dataset_id) is True
    assert await delete_affiliation(session, affil.affil_id) is True

    print("\n✅ 全部测试通过！")


async def run_all():
    engine = create_async_engine(DATABASE_URL, echo=True)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    # ---- 初始化数据库结构 ----
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("✅ 使用 async_engine 初始化数据库结构完成")

    # ---- 清空所有表数据 ----
    async with SessionLocal() as session:
        await clear_all_tables(lambda: session)

    # ---- 执行测试逻辑 ----
    async with SessionLocal() as session:
        # 你之前的 run_tests 中的所有测试逻辑写在这里即可
        await run_tests(session)

    # ---- 主动关闭引擎连接池 ----
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_all())