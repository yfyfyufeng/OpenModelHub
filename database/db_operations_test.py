import json
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

# ========= Load Input Data ==============

async def load_insert_record(session):

    # 读取 JSON 文件中的数据
    rec_path = 'records/db_operations_test_original.json'

    # todo: remove this and all following pritnings
    with open(rec_path, 'r') as f:
        print(f"Loading data from {rec_path}...")
        data = json.load(f)
        print("Data loaded successfully.")
        
        # 插入 Affiliation 数据
    for affil in data['affiliation']:
        affil_record = await create_affiliation(session, affil['affil_name'])

        # 插入 User 数据
    for user in data['user']:
        user_record = await create_user(session,user)

        # 插入 Dataset 数据
    for dataset in data['dataset']:
        dataset_record = await create_dataset(session, dataset)
        
        # todo: insert affiliation
        # await link_user_affiliatidataseton(session, dataset)
    
    # todo: read other types of tables
    
    # todo: remove debug
    
    flag = True
    for model in data['model']:
        model_record = await create_model(session, model)
        if flag:
            print("model:\n", model)
            print("model_record:\n",model_record)
            flag = False
        # await link_model_author(session, model_record.model_id, user_['user_id'])
        
    return data

# ========= Run All Tests =========
async def run_tests(session: AsyncSession):
    
    await load_insert_record(session)
    choice = input("Record creation is completed. Do you want to empty the dataset? y/n: ")
    if choice != 'y':
        print("\n✅ 所有测试已成功通过,数据库未清空。")
        return

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
