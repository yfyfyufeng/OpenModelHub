import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from database_interface import *

# ========= Load Env =========
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")

DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"

# ========= Run All Tests =========
async def run_tests(session: AsyncSession):
    # -----------------------------
    # 创建 Affiliation
    # -----------------------------
    affil = await create_affiliation(session, "OpenAI")
    assert affil.affil_name == "OpenAI"
    affil = await update_affiliation(session, affil.affil_id, {"affil_name": "OpenAI Research"})
    assert affil.affil_name == "OpenAI Research"

    # -----------------------------
    # 创建 User
    # -----------------------------
    user = await create_user(session, {"user_name": "Alice", "affiliate": "TestLab"})
    user_id = user.user_id
    await link_user_affiliation(session, user_id, affil.affil_id)
    user = await update_user(session, user_id, {"user_name": "Alice A."})
    assert user.user_name == "Alice A."

    # -----------------------------
    # 创建 Dataset
    # -----------------------------
    dataset_data = {
        "ds_name": "COCO",
        "ds_size": 50000,
        "media": "image",
        "task": 1,
        "columns":[
            {"col_name": "image", "col_datatype": "string"},
            {"col_name": "label", "col_datatype": "int"}
        ]
    }
    dataset = await create_dataset(session, dataset_data)
    dataset_id = dataset.ds_id
    await link_user_dataset(session, user_id, dataset_id)
    dataset = await update_dataset(session, dataset_id, {"ds_size": 55000})
    assert dataset.ds_size == 55000

    # -----------------------------
    # 创建 CNN 模型
    # -----------------------------
    cnn_model_data = {
        "model_name": "YOLOv7",
        "param_num": 64000000,
        "media_type": "image",
        "arch_name": ArchType.CNN,
        "task": ["Detection"],
        "module_num": 10,
        "modules": [
            {"conv_size": 32, "pool_type": "max"},
            {"conv_size": 64, "pool_type": "avg"}
        ]
    }
    cnn_model = await create_model(session, cnn_model_data)
    await link_model_author(session, cnn_model.model_id, user_id)
    await link_model_dataset(session, cnn_model.model_id, dataset_id)

    # -----------------------------
    # 创建 RNN 模型
    # -----------------------------
    rnn_model_data = {
        "model_name": "LSTM",
        "param_num": 22000000,
        "media_type": "text",
        "arch_name": ArchType.RNN,
        "task": ["Generation"],
        "criteria": "MSE",
        "batch_size": 32,
        "input_size": 256
    }
    rnn_model = await create_model(session, rnn_model_data)
    await link_model_author(session, rnn_model.model_id, user_id)

    # -----------------------------
    # 创建 Transformer 模型
    # -----------------------------
    transformer_model_data = {
        "model_name": "BERT",
        "param_num": 110000000,
        "media_type": "text",
        "arch_name": ArchType.TRANSFORMER,
        "task": ["Classification"],
        "decoder_num": 6,
        "attn_size": 512,
        "up_size": 2048,
        "down_size": 1024,
        "embed_size": 768
    }
    transformer_model = await create_model(session, transformer_model_data)
    await link_model_author(session, transformer_model.model_id, user_id)

    # -----------------------------
    # 删除模型，验证任务删除、关系删除、用户数据集不删
    # -----------------------------
    assert await delete_model(session, cnn_model.model_id) is True
    assert await delete_model(session, rnn_model.model_id) is True
    assert await delete_model(session, transformer_model.model_id) is True

    # 模型已删，任务记录应不存在
    tasks = await session.execute(select(ModelTask).where(ModelTask.model_id == cnn_model.model_id))
    assert tasks.scalar_one_or_none() is None

    # 用户和数据集应该依然存在
    user_check = await get_user_by_id(session, user_id)
    dataset_check = await get_dataset_by_id(session, dataset_id)
    assert user_check is not None
    assert dataset_check is not None

    # -----------------------------
    # 删除用户、数据集、机构
    # -----------------------------
    assert await delete_user(session, user_id) is True
    assert await delete_dataset(session, dataset_id) is True
    assert await delete_affiliation(session, affil.affil_id) is True

    print("\n✅ 所有测试已成功通过！")

async def run_all():
    # 1. 清空数据库（如果需要）
    await drop_database()

    # 2. 初始化数据库结构
    await init_database()

    # 3. 创建 async engine
    engine = create_async_engine(DATABASE_URL, echo=True)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    # 4. 执行测试逻辑
    async with SessionLocal() as session:
        await run_tests(session)

    # 5. 主动关闭引擎连接池
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_all())
