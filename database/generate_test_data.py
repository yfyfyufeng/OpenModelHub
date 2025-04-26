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

# Add project root to system path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root / "database")])
sys.path.extend([str(project_root), str(project_root / "frontend")])

from database.database_interface import (
    User, Model, Dataset, ModelTask, DsCol,
    init_database
)
from database.database_schema import Dataset_TASK, Media_type, Task_name, Trainname, ArchType
from frontend.db import get_db_session

# Model architecture types
ARCH_TYPES = [ArchType.CNN, ArchType.RNN, ArchType.TRANSFORMER]
# Media types
MEDIA_TYPES = [Media_type.TEXT, Media_type.IMAGE, Media_type.AUDIO, Media_type.VIDEO]
# Task types
TASK_TYPES = [Task_name.CLASSIFICATION, Task_name.DETECTION, Task_name.GENERATION, Task_name.SEGMENTATION]
# Training types
TRAIN_TYPES = [Trainname.PRETRAIN, Trainname.FINETUNE, Trainname.RL]


async def generate_users(session):
    """Generate test user data"""
    # Check if admin user already exists
    stmt = select(User).where(User.user_name == "admin")
    result = await session.execute(stmt)
    existing_admin = result.scalar_one_or_none()

    if existing_admin:
        print("Admin user already exists, skipping creation")
        users = [existing_admin]
    else:
        users = [
            User(
                user_name="admin",
                password_hash="admin",
                affiliate="System Administrator",
                is_admin=True
            )
        ]
        for user in users:
            session.add(user)
        await session.commit()

    # Add other test users
    test_users = [
        User(
            user_name="user1",
            password_hash="password1",
            affiliate="Tsinghua University",
            is_admin=False
        ),
        User(
            user_name="user2",
            password_hash="password2",
            affiliate="Peking University",
            is_admin=False
        )
    ]

    for user in test_users:
        # Check if user already exists
        stmt = select(User).where(User.user_name == user.user_name)
        result = await session.execute(stmt)
        if not result.scalar_one_or_none():
            session.add(user)

    await session.commit()
    return users + test_users


async def generate_models(session, admin_user):
    """Generate test model data"""
    models = []
    for i in range(30):  # Increase to 30 models
        # Create model
        model = Model(
            model_name=f"Test_Model_{i + 1}",
            arch_name=random.choice(ARCH_TYPES).value,
            param_num=random.randint(1000000, 100000000),
            media_type=random.choice(MEDIA_TYPES).value,
            trainname=random.choice(TRAIN_TYPES).value,
            param=random.randint(1, 100),
            creator_id=admin_user.user_id
        )
        session.add(model)
        await session.flush()

        # Add tasks to model, ensuring no duplicates
        num_tasks = random.randint(1, 4)
        used_tasks = set()
        for j in range(num_tasks):
            while True:
                task_name = random.choice(TASK_TYPES)
                if task_name not in used_tasks:
                    used_tasks.add(task_name)
                    break

            task = ModelTask(
                model_id=model.model_id,
                task_name=task_name.value
            )
            session.add(task)

        models.append(model)

    await session.commit()
    return models


async def generate_datasets(session, admin_user):
    """Generate test dataset data"""
    datasets = []
    for i in range(20):  # Increase to 20 datasets
        # Create dataset
        dataset = Dataset(
            ds_name=f"Test_Dataset_{i + 1}",
            media=random.choice(MEDIA_TYPES).value,
            ds_size=random.randint(1024, 1024 * 1024 * 10),
            creator_id=admin_user.user_id
        )
        session.add(dataset)
        await session.flush()

        # Create data columns
        num_columns = random.randint(5, 15)
        for j in range(num_columns):
            column = DsCol(
                ds_id=dataset.ds_id,
                col_name=f"column_{j + 1}",
                col_datatype=random.choice(["int", "float", "varchar(255)", "text", "datetime", "boolean"])
            )
            session.add(column)

        # Add tasks to dataset
        num_tasks = random.randint(1, 4)
        used_tasks = set()
        for j in range(num_tasks):
            while True:
                task_name = random.choice(TASK_TYPES)
                if task_name not in used_tasks:
                    used_tasks.add(task_name)
                    break

            task = Dataset_TASK(
                ds_id=dataset.ds_id,
                task=task_name.value
            )
            session.add(task)

        datasets.append(dataset)

    await session.commit()
    return datasets


async def delete_test_data(session):
    """Delete all test data"""
    try:
        # Delete all datasets (cascade delete tasks and columns)
        print("Deleting datasets...")
        datasets = await session.execute(select(Dataset))
        for dataset in datasets.scalars():
            await session.delete(dataset)

        # Delete all models (cascade delete tasks)
        print("Deleting models...")
        models = await session.execute(select(Model))
        for model in models.scalars():
            await session.delete(model)

        # Delete test users (keep admin user)
        print("Deleting test users...")
        users = await session.execute(select(User).where(User.user_name != "admin"))
        for user in users.scalars():
            await session.delete(user)

        await session.commit()
        print("All test data deleted!")

    except Exception as e:
        print(f"Error while deleting data: {str(e)}")
        await session.rollback()
        raise


async def convert_to_json(session, users, models, datasets):
    """Convert database objects to JSON format"""
    json_data = {
        "user": [],
        "model": [],
        "dataset": []
    }

    # Convert user data
    for user in users:
        json_data["user"].append({
            "user_name": user.user_name,
            "password_hash": user.password_hash,
            "affiliate": user.affiliate,
            "is_admin": user.is_admin
        })

    # Convert model data
    for model in models:
        stmt = select(ModelTask).where(ModelTask.model_id == model.model_id)
        result = await session.execute(stmt)
        tasks = result.scalars().all()

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

    # Convert dataset data
    for dataset in datasets:
        stmt = select(Dataset_TASK).where(Dataset_TASK.ds_id == dataset.ds_id)
        result = await session.execute(stmt)
        tasks = result.scalars().all()

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
    """Main function"""
    # Initialize database
    await init_database()

    # Get database session
    Session = get_db_session()
    async with Session() as session:
        # Decide to generate or delete data based on command-line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "delete":
            await delete_test_data(session)
        else:
            print("Generating user data...")
            users = await generate_users(session)
            print(f"{len(users)} users generated")

            admin_user = next((user for user in users if user.user_name == "admin"), None)
            if not admin_user:
                raise ValueError("Admin user not found")

            print("Generating model data...")
            models = await generate_models(session, admin_user)
            print(f"{len(models)} models generated")

            print("Generating dataset data...")
            datasets = await generate_datasets(session, admin_user)
            print(f"{len(datasets)} datasets generated")

            json_data = await convert_to_json(session, users, models, datasets)

            output_file = Path(__file__).parent / "records" / "generated_test_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

            print(f"Test data saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
