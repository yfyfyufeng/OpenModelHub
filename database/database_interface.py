from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import Sequence, Optional, Dict, Union, List
from database_schema import (
    Model, CNN, RNN, Transformer, ModelTask, ModelAuthor,
    Dataset, ModelDataset, Module, DsCol,
    User, UserDataset, UserAffil, Affil,  Base, ArchType
)
from sqlalchemy.orm import joinedload, subqueryload
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_schema import Base
import os
import aiomysql
from dotenv import load_dotenv
import asyncio
from sqlalchemy.orm import selectinload

# --------------------------------------
# 🔧 Model-related Operations
# --------------------------------------
# --------------------------------------
# 🔧 创建模型（按类型分发）
# --------------------------------------
async def create_model(session: AsyncSession, model_data: Dict) -> Union[Model, CNN, RNN, Transformer]:
    if not model_data.get("task"):
        raise ValueError("A model must have at least one task")

    model = Model(
        model_name=model_data["model_name"],
        param_num=model_data["param_num"],
        media_type=model_data["media_type"],
        arch_name=model_data["arch_name"],
    )
    session.add(model)
    await session.flush()

    for task_name in model_data["task"]:
        task = ModelTask(
            model_id=model.model_id,
            task_name=task_name
        )
        session.add(task)

    await session.commit()
    await session.refresh(model)

    arch_name = model_data["arch_name"]

    if arch_name == ArchType.CNN:
        cnn = CNN(model_id=model.model_id, module_num=model_data["module_num"])
        session.add(cnn)
        await session.commit()
        await session.refresh(cnn)

        for module_data in model_data.get("modules", []):
            module = Module(
                model_id=cnn.model_id,
                conv_size=module_data["conv_size"],
                pool_type=module_data["pool_type"]
            )
            session.add(module)

        await session.commit()
        await session.refresh(cnn)
        return cnn

    elif arch_name == ArchType.RNN:
        rnn = RNN(
            model_id=model.model_id,
            criteria=model_data["criteria"],
            batch_size=model_data["batch_size"],
            input_size=model_data["input_size"]
        )
        session.add(rnn)
        await session.commit()
        await session.refresh(rnn)
        return rnn

    elif arch_name == ArchType.TRANSFORMER:
        tf = Transformer(
            model_id=model.model_id,
            decoder_num=model_data["decoder_num"],
            attn_size=model_data["attn_size"],
            up_size=model_data["up_size"],
            down_size=model_data["down_size"],
            embed_size=model_data["embed_size"]
        )
        session.add(tf)
        await session.commit()
        await session.refresh(tf)
        return tf

    else:
        raise ValueError(f"Unsupported architecture type: {arch_name}")


# --------------------------------------
# 🔍 查询模型
# --------------------------------------
async def get_model_by_id(session: AsyncSession, model_id: int) -> Optional[Model]:
    result = await session.execute(
        select(Model)
        .options(
            selectinload(Model.tasks),
            selectinload(Model.authors),
            selectinload(Model.datasets),
            selectinload(Model.cnn),
            selectinload(Model.rnn),
            selectinload(Model.transformer),
        )
        .filter_by(model_id=model_id)
    )
    return result.scalar_one_or_none()


async def list_models(session: AsyncSession) -> Sequence[Model]:
    """获取所有模型"""
    stmt = select(Model).options(
        selectinload(Model.tasks),
        selectinload(Model.authors),
        selectinload(Model.datasets),
        selectinload(Model.cnn),
        selectinload(Model.rnn),
        selectinload(Model.transformer),
    )
    result = await session.execute(stmt)
    return result.scalars().all()


# --------------------------------------
# 🔧 更新模型
# --------------------------------------
async def update_model(session: AsyncSession, model_id: int, update_data: Dict) -> Optional[Model]:
    model = await get_model_by_id(session, model_id)
    if not model:
        return None

    for key, value in update_data.items():
        if hasattr(model, key):
            setattr(model, key, value)

    if "task" in update_data:
        await session.execute(delete(ModelTask).where(ModelTask.model_id == model_id))
        for task_name in update_data["task"]:
            task = ModelTask(model_id=model.model_id, task_name=task_name)
            session.add(task)

    if "authors" in update_data:
        await session.execute(delete(ModelAuthor).where(ModelAuthor.model_id == model_id))
        for author_id in update_data["authors"]:
            author = ModelAuthor(model_id=model.model_id, user_id=author_id)
            session.add(author)

    if "datasets" in update_data:
        await session.execute(delete(ModelDataset).where(ModelDataset.model_id == model_id))
        for dataset_id in update_data["datasets"]:
            dataset = ModelDataset(model_id=model.model_id, dataset_id=dataset_id)
            session.add(dataset)

    if model.arch_name == ArchType.CNN and "cnn" in update_data:
        cnn = model.cnn
        for key, value in update_data["cnn"].items():
            setattr(cnn, key, value)
    elif model.arch_name == ArchType.RNN and "rnn" in update_data:
        rnn = model.rnn
        for key, value in update_data["rnn"].items():
            setattr(rnn, key, value)
    elif model.arch_name == ArchType.TRANSFORMER and "transformer" in update_data:
        transformer = model.transformer
        for key, value in update_data["transformer"].items():
            setattr(transformer, key, value)

    await session.commit()
    await session.refresh(model)
    return model


# --------------------------------------
# ❌ 删除模型
# --------------------------------------
async def delete_model(session: AsyncSession, model_id: int) -> bool:
    await session.execute(delete(ModelTask).where(ModelTask.model_id == model_id))

    await session.execute(delete(ModelAuthor).where(ModelAuthor.model_id == model_id))

    await session.execute(delete(ModelDataset).where(ModelDataset.model_id == model_id))

    await session.execute(delete(Module).where(Module.model_id == model_id))
    await session.execute(delete(CNN).where(CNN.model_id == model_id))
    await session.execute(delete(RNN).where(RNN.model_id == model_id))
    await session.execute(delete(Transformer).where(Transformer.model_id == model_id))

    model = await get_model_by_id(session, model_id)
    if model:
        await session.delete(model)
        await session.commit()
        return True

    return False


# --------------------------------------
# 🔧 Dataset-related Operations
# --------------------------------------
async def create_dataset(session: AsyncSession, dataset_data: dict) -> Dataset:
    dataset = Dataset(
        ds_name=dataset_data["ds_name"],
        ds_size=dataset_data["ds_size"],
        media=dataset_data["media"],
        task=dataset_data["task"]
    )
    session.add(dataset)
    await session.flush()

    if not dataset_data['columns']:
        raise ValueError("A dataset must have at least one column")

    for col_data in dataset_data['columns']:
        ds_col = DsCol(
            ds_id=dataset.ds_id,
            col_name=col_data["col_name"],
            col_datatype=col_data["col_datatype"]
        )
        session.add(ds_col)

    await session.commit()
    await session.refresh(dataset)
    return dataset


async def get_dataset_by_id(session: AsyncSession, ds_id: int) -> Optional[Dataset]:
    result = await session.execute(
        select(Dataset)
        .options(selectinload(Dataset.columns))
        .filter_by(ds_id=ds_id)
    )
    return result.scalar_one_or_none()


async def list_datasets(session: AsyncSession) -> Sequence[Dataset]:
    """获取所有数据集"""
    stmt = select(Dataset).options(
        selectinload(Dataset.columns)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def delete_dataset(session: AsyncSession, ds_id: int) -> bool:
    await session.execute(delete(DsCol).where(DsCol.ds_id == ds_id))
    await session.execute(delete(UserDataset).where(UserDataset.ds_id == ds_id))
    await session.execute(delete(ModelDataset).where(ModelDataset.dataset_id == ds_id))

    dataset = await get_dataset_by_id(session, ds_id)
    if dataset:
        await session.delete(dataset)
        await session.commit()
        return True
    return False


async def update_dataset(session: AsyncSession, ds_id: int, update_data: dict) -> Optional[Dataset]:
    dataset = await get_dataset_by_id(session, ds_id)
    if dataset:
        for key, value in update_data.items():
            setattr(dataset, key, value)

        if 'columns' in update_data:
            for col_data in update_data['columns']:
                existing_col = await session.execute(
                    select(DsCol).filter_by(ds_id=ds_id, col_name=col_data["col_name"])
                )
                col = existing_col.scalar_one_or_none()
                if col:
                    col.col_datatype = col_data["col_datatype"]
                else:
                    ds_col = DsCol(
                        ds_id=ds_id,
                        col_name=col_data["col_name"],
                        col_datatype=col_data["col_datatype"]
                    )
                    session.add(ds_col)

        await session.commit()
        await session.refresh(dataset)
        return dataset
    return None


# --------------------------------------
# 🔧 User-related Operations
# --------------------------------------
async def create_user(session: AsyncSession, user_data: dict) -> User:
    existing = await session.execute(select(User).where(User.user_name == user_data["user_name"]))
    if existing.scalar_one_or_none():
        raise ValueError("用户名已存在")
    user = User(**user_data)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_by_id(session: AsyncSession, user_id: int) -> Optional[User]:
    result = await session.execute(select(User).filter_by(user_id=user_id))
    return result.scalar_one_or_none()


async def list_users(session: AsyncSession) -> Sequence[User]:
    result = await session.execute(select(User))
    return result.scalars().all()


async def delete_user(session: AsyncSession, user_id: int) -> bool:
    # 清理关联表
    await session.execute(delete(UserAffil).where(UserAffil.user_id == user_id))
    await session.execute(delete(UserDataset).where(UserDataset.user_id == user_id))
    await session.execute(delete(ModelAuthor).where(ModelAuthor.user_id == user_id))

    # 删除主表记录
    user = await get_user_by_id(session, user_id)
    if user:
        await session.delete(user)
        await session.commit()
        return True
    return False


async def update_user(session: AsyncSession, user_id: int, update_data: dict) -> Optional[User]:
    user = await get_user_by_id(session, user_id)
    if user:
        for key, value in update_data.items():
            setattr(user, key, value)
        await session.commit()
        await session.refresh(user)
        return user
    return None


# --------------------------------------
# 🔧 Affiliation Operations
# --------------------------------------
async def create_affiliation(session: AsyncSession, affil_name: str) -> Affil:
    affil = Affil(affil_name=affil_name)
    session.add(affil)
    await session.commit()
    await session.refresh(affil)
    return affil


async def list_affiliations(session: AsyncSession) -> Sequence[Affil]:
    result = await session.execute(select(Affil))
    return result.scalars().all()


async def get_affiliation_by_id(session: AsyncSession, affil_id: int) -> Optional[Affil]:
    result = await session.execute(select(Affil).filter_by(affil_id=affil_id))
    return result.scalar_one_or_none()


async def delete_affiliation(session: AsyncSession, affil_id: int) -> bool:
    await session.execute(delete(UserAffil).where(UserAffil.affil_id == affil_id))

    affil = await get_affiliation_by_id(session, affil_id)
    if affil:
        await session.delete(affil)
        await session.commit()
        return True
    return False


async def update_affiliation(session: AsyncSession, affil_id: int, update_data: dict) -> Optional[Affil]:
    affil = await get_affiliation_by_id(session, affil_id)
    if affil:
        for key, value in update_data.items():
            setattr(affil, key, value)
        await session.commit()
        await session.refresh(affil)
        return affil
    return None


# --------------------------------------
# 🔧 User-Affiliation Linking
# --------------------------------------
async def link_user_affiliation(session: AsyncSession, user_id: int, affil_id: int):
    relation = UserAffil(user_id=user_id, affil_id=affil_id)
    session.add(relation)
    await session.commit()


# --------------------------------------
# 🔧 Model-Dataset Linking
# --------------------------------------
async def link_model_dataset(session: AsyncSession, model_id: int, ds_id: int):
    link = ModelDataset(model_id=model_id, dataset_id=ds_id)
    session.add(link)
    await session.commit()


# --------------------------------------
# 🔧 Model-Author Linking
# --------------------------------------
async def link_model_author(session: AsyncSession, model_id: int, user_id: int):
    link = ModelAuthor(model_id=model_id, user_id=user_id)
    session.add(link)
    await session.commit()


# --------------------------------------
# 🔧 User-Dataset Linking
# --------------------------------------
async def link_user_dataset(session: AsyncSession, user_id: int, ds_id: int):
    link = UserDataset(user_id=user_id, ds_id=ds_id)
    session.add(link)
    await session.commit()


async def init_database():
    """初始化数据库"""
    load_dotenv()
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    TARGET_DB = os.getenv("TARGET_DB")
    
    # 异步连接 MySQL 默认数据库
    conn = await aiomysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        db='mysql'  # 确保连接默认库以便检查/创建目标库
    )

    async with conn.cursor() as cursor:
        await cursor.execute(f"SHOW DATABASES LIKE '{TARGET_DB}'")
        result = await cursor.fetchone()
        if not result:
            print(f"📦 数据库 `{TARGET_DB}` 不存在，正在创建...")
            await cursor.execute(
                f"CREATE DATABASE {TARGET_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            print(f"✅ 数据库 `{TARGET_DB}` 创建成功！")
        else:
            print(f"✅ 数据库 `{TARGET_DB}` 已存在")
            # 删除数据库并重新创建
            print(f"🔄 正在重新创建数据库 `{TARGET_DB}`...")
            await cursor.execute(f"DROP DATABASE {TARGET_DB};")
            await cursor.execute(
                f"CREATE DATABASE {TARGET_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            print(f"✅ 数据库 `{TARGET_DB}` 重新创建成功！")

    conn.close()

    # 初始化 SQLAlchemy 引擎（注意：SQLAlchemy 仍然是同步操作）
    db_url = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"
    engine = create_engine(db_url, echo=True)

    # 创建所有表
    Base.metadata.create_all(engine)

    # 创建 Session
    Session = sessionmaker(bind=engine)

    print("✅ 所有表结构已初始化完成")

    return Session()


async def drop_database():
    load_dotenv()
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", 3306))
    TARGET_DB = os.getenv("TARGET_DB")

    try:
        # 连接到默认的 mysql 数据库
        conn = await aiomysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            db='mysql'  # 确保连接到 'mysql' 而不是目标库
        )

        async with conn.cursor() as cursor:
            await cursor.execute(f"SHOW DATABASES LIKE '{TARGET_DB}'")
            result = await cursor.fetchone()

            if result:
                print(f"📦 数据库 `{TARGET_DB}` 存在，正在删除...")
                await cursor.execute(f"DROP DATABASE {TARGET_DB};")
                print(f"✅ 数据库 `{TARGET_DB}` 删除成功！")
            else:
                print(f"❌ 数据库 `{TARGET_DB}` 不存在，无法删除。")

        conn.close()
        await conn.wait_closed()

    except Exception as e:
        print(f"❌ 发生错误: {e}")


async def run_all():
    # 1. 清空数据库
    await drop_database()

    # 2. 初始化数据库结构
    await init_database()


if __name__ == "__main__":
    asyncio.run(run_all())
