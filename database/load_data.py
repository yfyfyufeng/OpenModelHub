import json
import os
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import shutil
# 添加项目根目录到系统路径
import sys
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

# 枚举值映射
TRAINNAME_MAP = {
    "pre-train": Trainname.PRETRAIN,
    "fine-tune": Trainname.FINETUNE,
    "finetune": Trainname.FINETUNE,  # 添加不带连字符的版本
    "rl": Trainname.RL
}

def patch_enum_fields(model: dict) -> dict:
    """将字符串类型的枚举值转换为对应的枚举类型"""
    # 处理 trainname
    trainname = model["trainname"].lower()
    # 移除可能的前缀
    if "." in trainname:
        trainname = trainname.split(".")[-1]
    # 标准化训练类型名称
    if trainname == "finetune":
        trainname = "fine-tune"
    if trainname in TRAINNAME_MAP:
        model["trainname"] = TRAINNAME_MAP[trainname]
    else:
        raise ValueError(f"无效的训练类型: {trainname}")
    
    # 处理 arch_name
    arch_name = model["arch_name"].upper()
    try:
        model["arch_name"] = ArchType[arch_name]  # 使用字典访问方式
    except KeyError:
        raise ValueError(f"无效的架构类型: {arch_name}")
    
    # 处理 media_type
    media_type = model["media_type"].upper()  # 直接转换为大写
    try:
        model["media_type"] = Media_type[media_type]  # 使用字典访问方式
    except KeyError:
        raise ValueError(f"无效的媒体类型: {media_type}")
    
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

async def load_json_file(session: AsyncSession, file_path: str, current_user: User = None):
    """加载单个 JSON 文件并插入数据"""
    print(f"正在加载文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # 插入 Affiliation 数据
        if 'affiliation' in data:
            for affil in data['affiliation']:
                await create_affiliation(session, affil['affil_name'])
        
        # 插入 User 数据
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
                    await create_user(session, user_data)
                else:
                    print(f"用户 {user['user_name']} 已存在，跳过创建")
        
        # 插入 Dataset 数据
        if 'dataset' in data:
            for dataset in data['dataset']:
                # 保存数据集文件
                if 'file_data' in dataset:
                    file_path = save_file(dataset['file_data'], f"{dataset['ds_name']}.txt")
                    dataset['file_path'] = file_path
                
                await create_dataset(session, dataset)
        
        # 插入 Model 数据
        if 'model' in data:
            for model in data['model']:
                # 保存模型文件
                if 'file_data' in model:
                    file_path = save_file(model['file_data'], f"{model['model_name']}.pt")
                    model['file_path'] = file_path
                
                try:
                    model = patch_enum_fields(model)
                    model = add_default_values(model)
                    await create_model(session, model)
                except ValueError as e:
                    print(f"处理模型 {model.get('model_name', '未知')} 时出错: {str(e)}")
                    continue
    
    print(f"文件 {file_path} 加载完成")

async def load_all_records(session: AsyncSession, current_user: User = None):
    """加载 records 目录下最新的 JSON 文件"""
    records_dir = Path(__file__).parent / "records"
    
    if not records_dir.exists():
        print(f"目录 {records_dir} 不存在")
        return
    
    # 获取所有JSON文件并按修改时间排序
    json_files = list(records_dir.glob("data_*.json"))
    if not json_files:
        print(f"在 {records_dir} 中没有找到数据文件")
        return
    
    # 按修改时间排序，获取最新的文件
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"找到最新的数据文件: {latest_file.name}")
    
    # 只加载最新的文件
    await load_json_file(session, str(latest_file), current_user)
    
    print("数据加载完成")

async def main():
    """主函数"""
    # 初始化数据库
    await init_database()
    
    # 获取数据库会话
    from frontend.db import get_db_session
    Session = get_db_session()
    
    async with Session() as session:
        await load_all_records(session)
        await session.commit()

if __name__ == "__main__":
    asyncio.run(main()) 