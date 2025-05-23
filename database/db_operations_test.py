import json
import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from database_interface import *
# todo: and other domains from db_schema
from database_schema import ArchType, Trainname, Media_type


# ========= Load Env =========
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")

DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"

# ========= Load Input Data ==============


def patch_enum_fields(model: dict) -> dict:
    model["trainname"] = Trainname(model["trainname"])
    model["arch_name"] = ArchType(model["arch_name"])
    model["media_type"] = Media_type(model["media_type"].lower())
    return model

async def load_insert_record(session):

    # 读取 JSON 文件中的数据
    parent_path = "records"
    # Find all JSON files in the specified path
    json_files = [f for f in os.listdir(parent_path) if f.endswith('.json')]

    # List the JSON files for the user to choose from
    print("Available JSON files:")
    for idx, file in enumerate(json_files):
        print(f"{idx + 1}: {file}")

    # Let the user input a number to choose a file
    while True:
        try:
            choice = int(input("Enter the number of the JSON file to load: "))
            if 1 <= choice <= len(json_files):
                rec_path = os.path.join(parent_path, json_files[choice - 1])
                break
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    rec_path = os.path.join(parent_path, json_files[choice - 1])

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
        # todo: insert affiliation link
        # await link_user_affiliatidataseton(session, dataset)
    
    # todo: read other types of tables
    for model in data['model']:
        model = patch_enum_fields(model)
        model_record = await create_model(session, model)
        # await link_model_author(session, model_record.model_id, user_['user_id'])
    
    return data

# ========= Run All Tests =========
async def run_tests(session: AsyncSession):
    
    await load_insert_record(session)
    # choice = input("Record creation is completed. Do you want to empty the dataset? y/n: ")
    choice = 'n'
    if choice != 'y':
        print("\n✅ All test has been passed. Database is not emptied.")
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