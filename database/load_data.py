import json
import enum
import os
import sys
import random
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import shutil
from database_interface import *
from database_schema import ArchType, Trainname, Media_type, Task_name
# 添加项目根目录到系统路径


current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
from database_interface import (
    User, Model, Dataset, ModelTask, DsCol,
    init_database, create_affiliation, create_user, create_dataset, create_model
)
from database_schema import ArchType, Trainname, Media_type, Task_name

# 创建数据存储目录
DATA_DIR = project_root / "database" / "data"
DATA_DIR.mkdir(exist_ok=True)

def save_file(file_data: bytes, filename: str) -> str:
    """保存文件到数据目录"""
    file_path = DATA_DIR / filename
    with open(file_path, "wb") as f:
        f.write(file_data)
    return str(file_path)

# Mapping string values to enum members

TRAINNAME_MAP = {
    "pre-train": Trainname.PRETRAIN,
    "fine-tune": Trainname.FINETUNE,
    "reinforcement learning": Trainname.RL
}

ARCHTYPE_MAP = {
    "cnn": ArchType.CNN,
    "rnn": ArchType.RNN,
    "transformer": ArchType.TRANSFORMER
}

MEDIATYPE_MAP = {
    "text": Media_type.TEXT,
    "image": Media_type.IMAGE,
    "audio": Media_type.AUDIO,
    "video": Media_type.VIDEO
}


TASKTYPE_SET = {t.value for t in Task_name}

def patch_enum_fields(model: dict) -> dict:
    """
    Convert string-based enum fields in the model dict to their enum values using mapping.
    Also validates task names against the enums.
    Can be applied to models and datasets.
    """

    # -------- Trainname --------
    if "trainname" in model:
        trainname = model.get("trainname", "").lower()
        if trainname in TRAINNAME_MAP:
            model["trainname"] = TRAINNAME_MAP[trainname]
        else:
            raise ValueError(f"❌ Invalid trainname: '{trainname}'. Expected one of: {list(TRAINNAME_MAP.keys())}")

    # -------- ArchType --------
    if "arch_name" in model:
        arch = model.get("arch_name", "").lower()
        if arch in ARCHTYPE_MAP:
            model["arch_name"] = ARCHTYPE_MAP[arch]
        else:
            raise ValueError(f"❌ Invalid architecture: '{arch}'. Expected one of: {list(ARCHTYPE_MAP.keys())}")

    # -------- MediaType --------
    if "media_type" in model:
        media = model.get("media_type", "").lower()
        if media in MEDIATYPE_MAP:
            model["media_type"] = MEDIATYPE_MAP[media]
        else:
            raise ValueError(f"❌ Invalid media type: '{media}'. Expected one of: {list(MEDIATYPE_MAP.keys())}")

    # -------- Task List --------
    if "task" in model:
        if isinstance(model["task"], list):
            validated_tasks = []
            for task in model["task"]:
                task_lower = task.lower()
                if task_lower in TASKTYPE_SET:
                    validated_tasks.append(task_lower)
                else:
                    raise ValueError(f"❌ Invalid task type: '{task}'. Expected one of: {list(TASKTYPE_SET)}")
            model["task"] = validated_tasks

    return model

def add_default_values(model: dict) -> dict:
    
    """为模型添加默认值"""
    defaults = {
        "criteria": "MSE",
        "batch_size": 32,
        "input_size": 256,
        "module_num": 10,
        "modules": [
            {"conv_size": 32, "pool_type": "max"},
            {"conv_size": 64, "pool_type": "avg"}
        ],
        "decoder_num": 6,
        "attn_size": 512,
        "up_size": 2048,
        "down_size": 1024,
        "embed_size": 768
    }
    
    # 只添加不存在的字段
    for key, value in defaults.items():
        if key not in model:
            model[key] = value
    
    return model

async def check_user_exists(session: AsyncSession, user_name: str) -> bool:
    
    """检查用户是否已存在"""
    stmt = select(User).where(User.user_name == user_name)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None

async def get_user_by_name(session: AsyncSession, user_name: str) -> User:
    """根据用户名获取用户"""
    stmt = select(User).where(User.user_name == user_name)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def extract_names(session: AsyncSession, file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f2:
        output_path = project_root / "database" / "data" / "names.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            data = json.load(f2)
            if "affiliation" in data:
                affil_names = [affil['affil_name'] for affil in data['affiliation']]
                f.write(f"affil_names:\n{affil_names}\n")
                f.write("\n")
            if "user" in data:
                user_names = [user['user_name'] for user in data['user']]
                f.write(f"user_names:\n{user_names}\n")
            if "model" in data:
                model_names = [model['model_name'] for model in data['model']]
                # f.write("\n".join(model_names))
                f.write(f'model_names:\n{model_names}\n')
            if "dataset" in data:
                dataset_names = [dataset['ds_name'] for dataset in data['dataset']]
                f.write(f'ds_names:\n{dataset_names}\n')
    

async def load_json_file(session: AsyncSession, file_path: str, current_user: User = None):
    """加载单个 JSON 文件并插入数据"""
    print(f"Loading file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # check whether relationship exists. if no, write it to json file.
        relationship_exist = False
        relationship_ls = ['user_ds','model_author','model_dataset']
        for rl in relationship_ls:
            if rl in data:
                relationship_exist = True
                break
        
        # I. instances
        
        # 1. Affiliation
        affil_recs = []
        if 'affiliation' in data:
            for affil in data['affiliation']:
                affil_record = await create_affiliation(session, affil['affil_name'])
                affil_recs.append(affil_record)
        
        affil_dic = {}
        for i in range(len(affil_recs)):
            affil_dic[affil_recs[i].affil_name] = affil_recs[i].affil_id
        
        user_recs = []
        # 2. User
        if 'user' in data:
            for user in data['user']:
                # 检查用户是否已存在
                if not await check_user_exists(session, user['user_name']):
                    # 确保用户数据包含密码
                    user_data = {
                        "user_name": user['user_name'],
                        "password_hash": user.get('password_hash'),  # 如果没有密码，使用默认密码
                        "affiliate": user.get('affiliate'),
                        "is_admin": user.get('is_admin', False)
                    }
                    user_record = await create_user(session, user_data)
                    user_recs.append(user_record)
                else:
                    # username is also a primary key, in this case.
                    print(f"用户 {user['user_name']} 已存在，跳过创建")
        # Relationshpi 1: user & affil
        for user in user_recs:
            if user.affiliate in affil_dic:
                await link_user_affiliation(session, user.user_id, affil_dic[user.affiliate])
            else:
                print(f"用户 {user.user_name}'s affiliate {user.affiliate} doesn't exist, skip linking.")
        
        
        # 3. Dataset
        ds_recs = []
        ds_media = {"text": [], "audio": [], "image": [],"video": []}
        if 'dataset' in data:
            for dataset in data['dataset']:
                # 保存数据集文件
                # if 'file_data' in dataset:
                #     file_path = save_file(dataset['file_data'], f"{dataset['ds_name']}.txt")
                #     dataset['file_path'] = file_path
                try:
                    dataset = patch_enum_fields(dataset)
                
                except ValueError as e:
                    print(f"Value error in dataset record {dataset}")
                
                ds_record = await create_dataset(session, dataset)
                ds_recs.append(ds_record)
                ds_media[dataset['media']].append(dataset['ds_name'])
                
        # 4. Model
        model_recs = []
        model_media = {"text": [], "audio": [], "image": [],"video": []}
        if 'model' in data:
            for model in data['model']:
                # # 保存模型文件
                # if 'file_data' in model:
                #     file_path = save_file(model['file_data'], f"{model['model_name']}.pt")
                #     model['file_path'] = file_path
                
                try:
                    model = patch_enum_fields(model)
                    model = add_default_values(model)
                    model_record = await create_model(session, model, retChild=False)
                    # for later building relationships
                    model_recs.append(model_record)
                    model_media[(model['media_type']).value].append(model['model_name'])
                except ValueError as e:
                    print(f"处理模型 {model.get('model_name', '未知')} 时出错: {str(e)}")
                    continue
                
        # II. relationships
        relationship_exist = False
        if relationship_exist:
            print("Relationship already exists, skipping random relationship creation...")
            if 'user_ds' in data:
                for user_ds in data['user_ds']:
                    user_id = user_ds['user_id']
                    ds_id = user_ds['ds_id']

                    user_obj = await session.get(User, user_id)
                    dataset_obj = await session.get(Dataset, ds_id)

                    if user_obj and dataset_obj:
                        await link_user_dataset(session, user_id, ds_id)
                    else:
                        print(f"⚠️ Skipping invalid user_ds link: user_id {user_id} or ds_id {ds_id} not found.")

            # 2. model & author
            if 'model_author' in data:
                for model_author in data['model_author']:
                    model_id = model_author['model_id']
                    user_id = model_author['user_id']

                    # 检查 model_id 和 user_id 是否存在
                    model_obj = await session.get(Model, model_id)
                    user_obj = await session.get(User, user_id)

                    if model_obj and user_obj:
                        await link_model_author(session, model_id, user_id)
                    else:
                        print(f"⚠️ Skipping invalid relationship: model_id {model_id} or user_id {user_id} not found.")

            # 3. model & dataset
            if 'model_dataset' in data:
                for model_dataset in data['model_dataset']:
                    model_id = model_dataset['model_id']
                    ds_id = model_dataset['ds_id']

                    # 检查 model 和 dataset 是否存在
                    model_obj = await session.get(Model, model_id)
                    dataset_obj = await session.get(Dataset, ds_id)

                    if model_obj and dataset_obj:
                        await link_model_dataset(session, model_id, ds_id)
                    else:
                        print(f"⚠️ Skipping invalid model_dataset link: model_id {model_id} or dataset_id {ds_id} not found.")

        else:
            # 1. random relationship: author and model/ds
            print("Relationship not exist, creating random relationships...")
            # 1.1. model
            model_author_recs = []
            for model in model_recs:
                if user_recs:
                    selected_authors = random.sample(user_recs, k=random.randint(1, min(4, len(user_recs))))
                    for author in selected_authors:
                        await link_model_author(session, model.model_id, author.user_id)
                        model_author_recs.append({"model_id": model.model_id, "user_id": author.user_id})
                else:
                    print(f"⚠️ No users available to assign as authors for model {model.model_name}")

            # 1.2. dataset

            user_ds_recs = []
            for dataset in ds_recs:
                if user_recs:
                    selected_authors = random.sample(user_recs, k=random.randint(1, min(4, len(user_recs))))
                    for author in selected_authors:
                        await link_user_dataset(session, author.user_id, dataset.ds_id)
                        user_ds_rec = {"user_id": author.user_id, "ds_id": dataset.ds_id}
                        user_ds_recs.append(user_ds_rec)
                        
                else:
                    print(f"⚠️ No users available to assign as authors for dataset {dataset.ds_name}")
            
            # 2. half-random
            model_dataset_recs = []
            for model in model_recs:
                
                matching_datasets = [dataset for dataset in ds_recs if dataset.media == model.media_type]
                
                if matching_datasets:
                    selected_datasets = random.sample(
                        matching_datasets,
                        k=random.randint(1, min(3, len(matching_datasets)))  # 每个模型连1~3个数据集
                    )
                    for dataset in selected_datasets:
                        await link_model_dataset(session, model.model_id, dataset.ds_id)
                        model_dataset_rec = {"model_id": model.model_id, "ds_id": dataset.ds_id}
                        model_dataset_recs.append(model_dataset_rec)
                else:
                    print(f"⚠️ No matching datasets for model {model.model_name} with media_type {model.media_type.value}")
                        
            # write to file
            data['user_ds'] = user_ds_recs
            data['model_author'] = model_author_recs
            data['model_dataset'] = model_dataset_recs
            # write to file

            write_to_file = False
            if write_to_file:
                output_path = file_path[:-5]+"_RELATIONSHIP.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        data,
                        f,
                        ensure_ascii=False,
                        indent=4,
                        default=lambda o: o.value if isinstance(o, enum.Enum) else str(o)
                    )
                print("Relationships written to file:", output_path)

    print(f"File {file_path} is loaded.")

async def load_all_records(session: AsyncSession, current_user: User = None):
    
    
    """加载 records 目录下最新的 JSON 文件"""
    
    records_dir = Path(__file__).parent / "records"
    
    if not records_dir.exists():
        print(f"Directory doesn't exist: {records_dir}.")
        return
    
    # 获取所有JSON文件并按修改时间排序
    json_files = list(records_dir.glob("*.json"))
    if not json_files:
        print(f"Find no json files in {records_dir}.")
        return
    
    # 按修改时间排序，获取最新的文件
    # latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("json files listed in chronological order (newest first):")
    
    for i in range(len(json_files)):
        
        print(f"{i+1}: {json_files[i].name}")
        
    # print(f"找到最新的数据文件: {latest_file.name}")
    
    choice = input("Please choose the numeber of file to load:\n> ")
    try:
        choice = int(choice)
        if choice < 1 or choice > len(json_files):
            print("Invalid input.")
            return
        chosen_file = json_files[choice - 1]
    except:
        print("Invalid input.")
        return

        
    # await extract_names(session, chosen_file)
    # return
    
    
    await load_json_file(session, str(chosen_file), current_user)    
    print("Finish loading json file.")

async def main():

    await drop_database()
    await init_database()
    
    # 获取数据库会话
    from frontend.db import get_db_session
    Session = get_db_session()
    
    async with Session() as session:
        await load_all_records(session)
        await session.commit()

if __name__ == "__main__":
    asyncio.run(main()) 