import asyncio
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sqlalchemy import select

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent.parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])

from database.database_interface import (
    User, Model, Dataset, ModelTask, DsCol,
    init_database
)
from frontend.db import get_db_session

# 模型架构类型
ARCH_TYPES = ["CNN", "RNN", "Transformer"]
# 媒体类型
MEDIA_TYPES = ["text", "image", "audio", "video"]
# 任务类型
TASK_TYPES = ["classification", "detection", "generation", "segmentation"]

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

async def generate_models(session):
    """生成测试模型数据"""
    models = []
    for i in range(10):
        # 创建模型
        model = Model(
            model_name=f"测试模型_{i+1}",
            arch_name=random.choice(ARCH_TYPES),
            param_num=random.randint(1000000, 10000000),
            media_type=random.choice(MEDIA_TYPES)
        )
        session.add(model)
        await session.flush()
        
        # 为模型添加任务，确保任务不重复
        num_tasks = random.randint(1, 3)
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
                task_name=task_name
            )
            session.add(task)
        
        models.append(model)
    
    await session.commit()
    return models

async def generate_datasets(session):
    """生成测试数据集数据"""
    datasets = []
    for i in range(5):
        # 创建数据集
        dataset = Dataset(
            ds_name=f"测试数据集_{i+1}",
            media=random.choice(MEDIA_TYPES),
            task=random.randint(1, 4),  # 使用整数表示任务类型
            ds_size=random.randint(1024, 1024*1024)  # 1KB to 1MB
        )
        session.add(dataset)
        await session.flush()
        
        # 创建数据列
        num_columns = random.randint(3, 10)
        for j in range(num_columns):
            column = DsCol(
                ds_id=dataset.ds_id,
                col_name=f"column_{j+1}",
                col_datatype=random.choice(["int", "float", "varchar(255)", "text"])
            )
            session.add(column)
        
        datasets.append(dataset)
    
    await session.commit()
    return datasets

async def main():
    """主函数"""
    # 初始化数据库
    await init_database()
    
    # 获取数据库会话
    Session = get_db_session()
    async with Session() as session:
        # 生成测试数据
        print("生成用户数据...")
        users = await generate_users(session)
        print(f"已生成 {len(users)} 个用户")
        
        print("生成模型数据...")
        models = await generate_models(session)
        print(f"已生成 {len(models)} 个模型")
        
        print("生成数据集数据...")
        datasets = await generate_datasets(session)
        print(f"已生成 {len(datasets)} 个数据集")
        
        print("测试数据生成完成！")

if __name__ == "__main__":
    asyncio.run(main()) 