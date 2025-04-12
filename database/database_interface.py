from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from typing import Sequence, Optional, Dict, Union
from database_schema import (
    Model, CNN, RNN, Transformer, ModelTask, ModelAuthor,
    Dataset, ModelDataset, Module, DsCol,
    User, UserDataset, UserAffil, Affil, text,  Base, ArchType
)
from sqlalchemy.orm import joinedload

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
            joinedload(Model.tasks),
            joinedload(Model.authors),
            joinedload(Model.datasets),
            joinedload(Model.cnn),
            joinedload(Model.rnn),
            joinedload(Model.transformer),
        )
        .filter_by(model_id=model_id)
    )
    return result.scalar_one_or_none()


async def list_models(session: AsyncSession) -> Sequence[Model]:
    result = await session.execute(
        select(Model)
        .options(
            joinedload(Model.tasks),
            joinedload(Model.authors),
            joinedload(Model.datasets),
            joinedload(Model.cnn),
            joinedload(Model.rnn),
            joinedload(Model.transformer),
        )
    )
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
    # ✅ 直接删除任务（ModelTask 是任务本体）
    await session.execute(delete(ModelTask).where(ModelTask.model_id == model_id))

    # ✅ 删除模型与作者的关联（不删 User）
    await session.execute(delete(ModelAuthor).where(ModelAuthor.model_id == model_id))

    # ✅ 删除模型与数据集的关联（不删 Dataset）
    await session.execute(delete(ModelDataset).where(ModelDataset.model_id == model_id))

    # ✅ 删除子架构信息
    await session.execute(delete(CNN).where(CNN.model_id == model_id))
    await session.execute(delete(RNN).where(RNN.model_id == model_id))
    await session.execute(delete(Transformer).where(Transformer.model_id == model_id))

    # ✅ 删除模型本体
    model = await get_model_by_id(session, model_id)
    if model:
        await session.delete(model)
        await session.commit()
        return True

    return False

# --------------------------------------
# 🔧 Dataset-related Operations
# --------------------------------------
# 创建数据集并添加列的函数
async def create_dataset(session: AsyncSession, dataset_data: dict, columns_data: list) -> Dataset:
    dataset = Dataset(
        ds_name=dataset_data["ds_name"],
        ds_size=dataset_data["ds_size"],
        media=dataset_data["media"],
        task=dataset_data["task"]
    )
    session.add(dataset)
    await session.flush()

    if not columns_data:
        raise ValueError("A dataset must have at least one column")

    for col_data in columns_data:
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
    result = await session.execute(select(Dataset).filter_by(ds_id=ds_id))
    return result.scalar_one_or_none()

async def list_datasets(session: AsyncSession) -> Sequence[Dataset]:
    result = await session.execute(select(Dataset))
    return result.scalars().all()

async def delete_dataset(session: AsyncSession, ds_id: int) -> bool:
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
        await session.commit()
        await session.refresh(dataset)
        return dataset
    return None

# --------------------------------------
# 🔧 User-related Operations
# --------------------------------------
async def create_user(session: AsyncSession, user_data: dict) -> User:
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
# 🔧 Task Operations
# --------------------------------------
async def add_task_to_model(session: AsyncSession, model_id: int, task_name: str):
    task = ModelTask(model_id=model_id, task_name=task_name)
    session.add(task)
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

# ========= Clear All Tables =========
async def clear_all_tables(get_session):
    """
    get_session: lambda or async function that returns an AsyncSession
    """
    async with get_session() as session:
        await session.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
        await session.commit()
        print("🧹 所有表数据已清空")
