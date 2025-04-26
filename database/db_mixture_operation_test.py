import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from database_interface import (
    create_affiliation, create_user, link_user_affiliation,
    create_dataset, link_user_dataset, create_model,
    link_model_author, link_model_dataset,
    get_model_info, get_dataset_info, get_user_info,
    get_model_ids_by_attribute, get_dataset_ids_by_attribute, get_user_ids_by_attribute,
    drop_database, init_database,
    ArchType, Trainname
)

# ========= Load Env =========
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")
DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"


# ========= 查询测试 =========
async def run_query_tests(session: AsyncSession):
    print("\n🚀 Start running query tests...\n")

    # 创建数据：Affiliation、User、Dataset、Model
    affil = await create_affiliation(session, "QueryAffil")
    user = await create_user(session, {"user_name": "QueryUser", "affiliate": "QueryLab"})
    await link_user_affiliation(session, user.user_id, affil.affil_id)

    dataset = await create_dataset(session, {
        "ds_name": "QueryDataset",
        "ds_size": 1234,
        "media": "text",
        "task": ["classification"],
        "columns": [{"col_name": "sentence", "col_datatype": "string"}]
    })
    await link_user_dataset(session, user.user_id, dataset.ds_id)

    model = await create_model(session, {
        "model_name": "QueryRNN",
        "param_num": 123456,
        "media_type": "text",
        "arch_name": ArchType.RNN,
        "trainname": Trainname.FINETUNE,
        "task": ["classification"],
        "criteria": "NLLLoss",
        "batch_size": 32,
        "input_size": 256,
        "param": 1
    })
    await link_model_author(session, model.model_id, user.user_id)
    await link_model_dataset(session, model.model_id, dataset.ds_id)

    # 查询模型信息
    model_info = await get_model_info(session, model.model_id)
    print("\n🔍 Model Info:")
    print(model_info)
    assert model_info["model_name"] == "QueryRNN"
    assert model_info["arch_name"] == "RNN"

    # 查询数据集信息
    dataset_info = await get_dataset_info(session, dataset.ds_id)
    print("\n📦 Dataset Info:")
    print(dataset_info)
    assert dataset_info["dataset"]["ds_name"] == "QueryDataset"
    assert dataset_info["columns"][0]["col_name"] == "sentence"

    # 查询用户信息
    user_info = await get_user_info(session, user.user_id)
    print("\n👤 User Info:")
    print(user_info)
    assert user_info["user"]["user_name"] == "QueryUser"
    assert "QueryDataset" in user_info["datasets"]
    assert "QueryRNN" in user_info["models"]

    # 字段查询：model
    model_ids = await get_model_ids_by_attribute(session, "model_name", "QueryRNN")
    print(f"\n🔎 model_ids by name = 'QueryRNN': {model_ids}")
    assert model.model_id in model_ids

    # 字段查询：dataset
    dataset_ids = await get_dataset_ids_by_attribute(session, "ds_name", "QueryDataset")
    print(f"\n🔎 dataset_ids by name = 'QueryDataset': {dataset_ids}")
    assert dataset.ds_id in dataset_ids

    # 字段查询：user
    user_ids = await get_user_ids_by_attribute(session, "user_name", "QueryUser")
    print(f"\n🔎 user_ids by name = 'QueryUser': {user_ids}")
    assert user.user_id in user_ids

    print("\n✅ All query interfaces has been tested!")


# ========= MAIN RUNNER =========
async def run_all():
    await drop_database()
    await init_database()

    engine = create_async_engine(DATABASE_URL, echo=True)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        await run_query_tests(session)

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_all())
