import json
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
    """从JSON文件加载数据并插入数据库"""
    try:
        # 使用UTF-8编码打开文件
        with open('database/records/data_20250424_204148.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 插入机构数据
        for affil_data in data.get('affiliation', []):
            affil = Affil(affil_name=affil_data['affil_name'])
            session.add(affil)
        
        # 插入用户数据
        users = {}  # 用于存储用户ID映射
        for user_data in data.get('user', []):
            user = User(
                user_name=user_data['user_name'],
                affiliate=user_data.get('affiliate'),
                is_admin=False
            )
            session.add(user)
            await session.flush()
            users[user.user_name] = user.user_id
        
        # 如果没有用户数据，创建一个默认管理员用户
        if not users:
            admin_user = User(
                user_name="admin",
                is_admin=True
            )
            session.add(admin_user)
            await session.flush()
            users["admin"] = admin_user.user_id
        
        # 获取默认创建者ID
        default_creator_id = next(iter(users.values()))
        
        # 插入数据集数据
        for dataset_data in data.get('dataset', []):
            dataset = Dataset(
                ds_name=dataset_data['ds_name'],
                ds_size=dataset_data['ds_size'],
                media=dataset_data['media'],
                description=dataset_data.get('description', ''),
                creator_id=default_creator_id  # 添加创建者ID
            )
            session.add(dataset)
            await session.flush()
            
            # 添加数据集列
            for col_data in dataset_data.get('columns', []):
                col = DsCol(
                    ds_id=dataset.ds_id,
                    col_name=col_data['col_name'],
                    col_datatype=col_data['col_datatype']
                )
                session.add(col)
            
            # 添加数据集任务
            for task_name in dataset_data.get('task', []):
                task = Dataset_TASK(
                    ds_id=dataset.ds_id,
                    task=task_name
                )
                session.add(task)
        
        # 插入模型数据
        for model_data in data.get('model', []):
            model = Model(
                model_name=model_data['model_name'],
                param_num=model_data['param_num'],
                media_type=model_data['media_type'],
                arch_name=model_data['arch_name'],
                trainname=model_data['trainname'],
                param=model_data.get('param', b''),
                creator_id=default_creator_id  # 添加创建者ID
            )
            session.add(model)
            await session.flush()
            
            # 添加模型任务
            for task_name in model_data.get('task', []):
                task = ModelTask(
                    model_id=model.model_id,
                    task_name=task_name
                )
                session.add(task)
            
            # 根据架构类型添加具体模型信息
            if model_data['arch_name'] == 'CNN':
                cnn = CNN(
                    model_id=model.model_id,
                    module_num=model_data.get('module_num', 0)
                )
                session.add(cnn)
                await session.flush()
                
                # 添加CNN模块
                for module_data in model_data.get('modules', []):
                    module = Module(
                        cnn_id=cnn.cnn_id,
                        conv_size=module_data['conv_size'],
                        pool_type=module_data['pool_type']
                    )
                    session.add(module)
            
            elif model_data['arch_name'] == 'RNN':
                rnn = RNN(
                    model_id=model.model_id,
                    criteria=model_data.get('criteria', ''),
                    batch_size=model_data.get('batch_size', 0),
                    input_size=model_data.get('input_size', 0)
                )
                session.add(rnn)
            
            elif model_data['arch_name'] == 'Transformer':
                transformer = Transformer(
                    model_id=model.model_id,
                    decoder_num=model_data.get('decoder_num', 0),
                    attn_size=model_data.get('attn_size', 0),
                    up_size=model_data.get('up_size', 0),
                    down_size=model_data.get('down_size', 0),
                    embed_size=model_data.get('embed_size', 0)
                )
                session.add(transformer)
        
        await session.commit()
        print("✅ 数据导入完成")
    except Exception as e:
        await session.rollback()
        print(f"❌ 数据导入失败: {str(e)}")
        raise e

# ========= Run All Tests =========
async def run_tests(session: AsyncSession):
    
    await load_insert_record(session)
    # choice = input("Record creation is completed. Do you want to empty the dataset? y/n: ")
    choice = 'n'
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
