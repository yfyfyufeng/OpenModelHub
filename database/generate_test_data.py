import asyncio
import os
import random
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import select

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])

from database.database_interface import (
    User, Model, Dataset, ModelTask, DsCol,
    init_database
)
from database.database_schema import Dataset_TASK, Media_type, Task_name, Trainname, ArchType
from frontend.db import get_db_session 

# 模型架构类型
ARCH_TYPES = [ArchType.CNN, ArchType.RNN, ArchType.TRANSFORMER]
# 媒体类型
MEDIA_TYPES = [Media_type.TEXT, Media_type.IMAGE, Media_type.AUDIO, Media_type.VIDEO]
# 任务类型
TASK_TYPES = [Task_name.CLASSIFICATION, Task_name.DETECTION, Task_name.GENERATION, Task_name.SEGMENTATION]
# 训练类型
TRAIN_TYPES = [Trainname.PRETRAIN, Trainname.FINETUNE, Trainname.RL]

async def generate_users(session):
    """生成测试用户数据"""
    # 检查admin用户是否已存在
    stmt = select(User).where(User.user_name == "admin")
    result = await session.execute(stmt)
    existing_admin = result.scalar_one_or_none()
    
    if existing_admin:
        print("管理员用户已存在，跳过创建")
        users = [existing_admin]
    else:
        users = [
            User(
                user_name="admin",
                password_hash="admin",
                affiliate="系统管理员",
                is_admin=True
            )
        ]
        for user in users:
            session.add(user)
        await session.commit()
    
    # 添加其他测试用户
    test_users = [
        User(
            user_name="user1",
            password_hash="password1",
            affiliate="清华大学",
            is_admin=False
        ),
        User(
            user_name="user2",
            password_hash="password2",
            affiliate="北京大学",
            is_admin=False
        )
    ]
    
    for user in test_users:
        # 检查用户是否已存在
        stmt = select(User).where(User.user_name == user.user_name)
        result = await session.execute(stmt)
        if not result.scalar_one_or_none():
            session.add(user)
    
    await session.commit()
    return users + test_users

async def generate_models(session, admin_user):
    """生成测试模型数据"""
    models = []
    for i in range(30):  # 增加到30个模型
        # 创建模型
        model = Model(
            model_name=f"测试模型_{i+1}",
            arch_name=random.choice(ARCH_TYPES).value,  # 使用枚举值
            param_num=random.randint(1000000, 100000000),  # 增加参数范围
            media_type=random.choice(MEDIA_TYPES).value,  # 使用枚举值
            trainname=random.choice(TRAIN_TYPES).value,  # 使用枚举值
            param=random.randint(1, 100),  # 添加param属性
            creator_id=admin_user.user_id  # 设置创建者为admin
        )
        session.add(model)
        await session.flush()
        
        # 为模型添加任务，确保任务不重复
        num_tasks = random.randint(1, 4)  # 增加最大任务数
        used_tasks = set()  # 用于跟踪已使用的任务
        for j in range(num_tasks):
            # 随机选择任务，直到找到一个未使用的任务
            while True:
                task_name = random.choice(TASK_TYPES)
                if task_name not in used_tasks:
                    used_tasks.add(task_name)
                    break
            
            task = ModelTask(
                model_id=model.model_id,
                task_name=task_name.value  # 使用枚举值
            )
            session.add(task)
        
        models.append(model)
    
    await session.commit()
    return models

async def generate_datasets(session, admin_user):
    """生成测试数据集数据"""
    datasets = []
    for i in range(20):  # 增加到20个数据集
        # 创建数据集
        dataset = Dataset(
            ds_name=f"测试数据集_{i+1}",
            media=random.choice(MEDIA_TYPES).value,  # 使用枚举值
            ds_size=random.randint(1024, 1024*1024*10),  # 增加数据集大小范围到10MB
            creator_id=admin_user.user_id  # 设置创建者为admin
        )
        session.add(dataset)
        await session.flush()
        
        # 创建数据列
        num_columns = random.randint(5, 15)  # 增加列数范围
        for j in range(num_columns):
            column = DsCol(
                ds_id=dataset.ds_id,
                col_name=f"column_{j+1}",
                col_datatype=random.choice(["int", "float", "varchar(255)", "text", "datetime", "boolean"])  # 增加数据类型
            )
            session.add(column)
        
        # 为数据集添加任务
        num_tasks = random.randint(1, 4)  # 增加最大任务数
        used_tasks = set()
        for j in range(num_tasks):
            while True:
                task_name = random.choice(TASK_TYPES)
                if task_name not in used_tasks:
                    used_tasks.add(task_name)
                    break
            
            task = Dataset_TASK(
                ds_id=dataset.ds_id,
                task=task_name.value  # 使用枚举值
            )
            session.add(task)
        
        datasets.append(dataset)
    
    await session.commit()
    return datasets

async def delete_test_data(session):
    """删除所有测试数据"""
    try:
        # 删除所有数据集（会级联删除相关的任务和列）
        print("删除数据集...")
        datasets = await session.execute(select(Dataset))
        for dataset in datasets.scalars():
            await session.delete(dataset)
        
        # 删除所有模型（会级联删除相关的任务）
        print("删除模型...")
        models = await session.execute(select(Model))
        for model in models.scalars():
            await session.delete(model)
        
        # 删除测试用户（保留admin用户）
        print("删除测试用户...")
        users = await session.execute(select(User).where(User.user_name != "admin"))
        for user in users.scalars():
            await session.delete(user)
        
        await session.commit()
        print("所有测试数据已删除！")
        
    except Exception as e:
        print(f"删除数据时出错: {str(e)}")
        await session.rollback()
        raise

async def convert_to_json(session, users, models, datasets):
    """将数据库对象转换为JSON格式"""
    json_data = {
        "user": [],
        "model": [],
        "dataset": []
    }
    
    # 转换用户数据
    for user in users:
        json_data["user"].append({
            "user_name": user.user_name,
            "password_hash": user.password_hash,
            "affiliate": user.affiliate,
            "is_admin": user.is_admin
        })
    
    # 转换模型数据
    for model in models:
        # 获取模型的任务
        stmt = select(ModelTask).where(ModelTask.model_id == model.model_id)
        result = await session.execute(stmt)
        tasks = result.scalars().all()
        
        # 获取创建者信息
        stmt = select(User).where(User.user_id == model.creator_id)
        result = await session.execute(stmt)
        creator = result.scalar_one_or_none()
        
        model_data = {
            "model_name": model.model_name,
            "arch_name": model.arch_name.value if hasattr(model.arch_name, 'value') else model.arch_name,
            "param_num": model.param_num,
            "media_type": model.media_type.value if hasattr(model.media_type, 'value') else model.media_type,
            "trainname": model.trainname.value if hasattr(model.trainname, 'value') else model.trainname,
            "param": model.param,
            "task": [task.task_name.value if hasattr(task.task_name, 'value') else task.task_name for task in tasks],
            "creator": creator.user_name if creator else None
        }
        
        json_data["model"].append(model_data)
    
    # 转换数据集数据
    for dataset in datasets:
        # 获取数据集的任务
        stmt = select(Dataset_TASK).where(Dataset_TASK.ds_id == dataset.ds_id)
        result = await session.execute(stmt)
        tasks = result.scalars().all()
        
        # 获取创建者信息
        stmt = select(User).where(User.user_id == dataset.creator_id)
        result = await session.execute(stmt)
        creator = result.scalar_one_or_none()
        
        dataset_data = {
            "ds_name": dataset.ds_name,
            "media": dataset.media.value if hasattr(dataset.media, 'value') else dataset.media,
            "ds_size": dataset.ds_size,
            "task": [task.task.value if hasattr(task.task, 'value') else task.task for task in tasks],
            "creator": creator.user_name if creator else None
        }
        
        json_data["dataset"].append(dataset_data)
    
    return json_data

async def main():
    """主函数"""
    # 初始化数据库
    await init_database()
    
    # 获取数据库会话
    Session = get_db_session()
    async with Session() as session:
        # 根据命令行参数决定是生成还是删除数据
        if len(sys.argv) > 1 and sys.argv[1] == "delete":
            # 删除测试数据
            await delete_test_data(session)
        else:
            # 生成测试数据
            print("生成用户数据...")
            users = await generate_users(session)
            print(f"已生成 {len(users)} 个用户")
            
            # 获取admin用户
            admin_user = next((user for user in users if user.user_name == "admin"), None)
            if not admin_user:
                raise ValueError("未找到admin用户")
            
            print("生成模型数据...")
            models = await generate_models(session, admin_user)
            print(f"已生成 {len(models)} 个模型")
            
            print("生成数据集数据...")
            datasets = await generate_datasets(session, admin_user)
            print(f"已生成 {len(datasets)} 个数据集")
            
            # 将数据转换为JSON格式
            json_data = await convert_to_json(session, users, models, datasets)
            
            # 保存JSON文件
            output_file = Path(__file__).parent / "records" / "generated_test_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            
            print(f"测试数据已保存到: {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 