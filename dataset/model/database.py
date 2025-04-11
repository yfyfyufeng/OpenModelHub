import pymysql
import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Enum, ForeignKey, text
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.dialects.mysql import BIGINT
from dotenv import load_dotenv
import enum

# ========= Load Env =========
load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
TARGET_DB = os.getenv("TARGET_DB", "openmodelhub")

# ========= Create Database =========
def create_database_if_not_exists():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database='mysql'
    )
    cursor = conn.cursor()
    cursor.execute(f"SHOW DATABASES LIKE '{TARGET_DB}'")
    result = cursor.fetchone()
    if not result:
        print(f"📦 数据库 `{TARGET_DB}` 不存在，正在创建...")
        cursor.execute(f"CREATE DATABASE {TARGET_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        print(f"✅ 数据库 `{TARGET_DB}` 创建成功！")
    else:
        print(f"✅ 数据库 `{TARGET_DB}` 已存在")
    cursor.close()
    conn.close()

# ========= SQLAlchemy Engine =========
def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}?charset=utf8mb4",
        echo=True
    )

Base = declarative_base()
Session = sessionmaker()

# ========= Tables =========

# ---------------------------
# 模型表
# ---------------------------
class ArchType(enum.Enum):
    CNN = "CNN"
    RNN = "RNN"
    TRANSFORMER = "Transformer"

class Model(Base):
    __tablename__ = 'model'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50))
    param_num = Column(BIGINT(unsigned=True))
    media_type = Column(String(50))
    arch_name = Column(Enum(ArchType), nullable=False)

    tasks = relationship("ModelTask", back_populates="model")
    authors = relationship("ModelAuthor", back_populates="model")
    datasets = relationship("ModelDataset", back_populates="model")

    cnn = relationship("CNN", uselist=False, back_populates="model")
    rnn = relationship("RNN", uselist=False, back_populates="model")
    transformer = relationship("Transformer", uselist=False, back_populates="model")

class ModelTask(Base):
    __tablename__ = 'model_tasks'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    task_name = Column(String(50), primary_key=True)

    model = relationship("Model", back_populates="tasks")

# ---------------------------
# Transformer 模型（子表）
# ---------------------------
class Transformer(Base):
    __tablename__ = 'transformer'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    decoder_num = Column(Integer)
    attn_size = Column(Integer)
    up_size = Column(Integer)
    down_size = Column(Integer)
    embed_size = Column(Integer)

    model = relationship("Model", back_populates="transformer")

# ---------------------------
# CNN 模型（子表）
# ---------------------------
class CNN(Base):
    __tablename__ = 'cnn'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    module_num = Column(Integer)

    model = relationship("Model", back_populates="cnn")
    modules = relationship("ModuleID", back_populates="cnn")


class ModuleID(Base):
    __tablename__ = 'module_id'

    model_id = Column(Integer, ForeignKey("cnn.model_id"), primary_key=True)
    conv_size = Column(Integer)
    pool_type = Column(String(20))

    cnn = relationship("CNN", back_populates="modules")

# ---------------------------
# RNN 模型（子表）
# ---------------------------
class RNN(Base):
    __tablename__ = 'rnn'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    criteria = Column(String(50))
    batch_size = Column(Integer)
    input_size = Column(Integer)

    model = relationship("Model", back_populates="rnn")

# ---------------------------
# 数据集表（Dataset）
# ---------------------------
class Dataset(Base):
    __tablename__ = 'dataset'

    ds_id = Column(Integer, primary_key=True, autoincrement=True)
    ds_name = Column(String(100))
    ds_size = Column(Integer)
    media = Column(String(50))
    task = Column(Integer)

    models = relationship("ModelDataset", back_populates="dataset")
    columns = relationship("DsCol", back_populates="dataset")
    users = relationship("UserDataset", back_populates="dataset")


class DsCol(Base):
    __tablename__ = 'ds_col'

    ds_col_id = Column(Integer, primary_key=True, autoincrement=True)
    ds_id = Column(Integer, ForeignKey("dataset.ds_id"))
    col_name = Column(String(50))
    col_datatype = Column(String(20))

    dataset = relationship("Dataset", back_populates="columns")

# ---------------------------
# 用户表(User)
# ---------------------------
class User(Base):
    __tablename__ = 'user'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(50))
    affiliate = Column(String(50))

    models = relationship("ModelAuthor", back_populates="user")
    datasets = relationship("UserDataset", back_populates="user")
    affiliations = relationship("UserAffil", back_populates="user")

# ---------------------------
# 机构表（Affil）
# ---------------------------
class Affil(Base):
    __tablename__ = 'affil'

    affil_id = Column(Integer, primary_key=True, autoincrement=True)
    affil_name = Column(String(100))

    users = relationship("UserAffil", back_populates="affiliation")

# ---------------------------
# 用户-机构 多对多中间表
# ---------------------------
class UserAffil(Base):
    __tablename__ = 'user_affil'

    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)
    affil_id = Column(Integer, ForeignKey("affil.affil_id"), primary_key=True)

    user = relationship("User", back_populates="affiliations")
    affiliation = relationship("Affil", back_populates="users")

# ---------------------------
# 用户-数据集 多对多中间表
# ---------------------------
class UserDataset(Base):
    __tablename__ = 'user_ds'

    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)
    ds_id = Column(Integer, ForeignKey("dataset.ds_id"), primary_key=True)

    user = relationship("User", back_populates="datasets")
    dataset = relationship("Dataset", back_populates="users")

# ---------------------------
# 模型-作者 多对多中间表
# ---------------------------
class ModelAuthor(Base):
    __tablename__ = 'model_author'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), primary_key=True)

    model = relationship("Model", back_populates="authors")
    user = relationship("User", back_populates="models")

# ---------------------------
# 模型-数据集 多对多中间表
# ---------------------------
class ModelDataset(Base):
    __tablename__ = 'model_dataset'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    dataset_id = Column(Integer, ForeignKey("dataset.dataset_id"), primary_key=True)

    model = relationship("Model", back_populates="datasets")
    dataset = relationship("Dataset", back_populates="models")

# ========= Init DB =========
def init_db():
    create_database_if_not_exists()
    engine = get_engine()
    Base.metadata.create_all(engine)
    Session.configure(bind=engine)
    print("\u2705 所有表结构已初始化完成")
    return Session()

# ========= Optional Clear for Test =========
async def clear_all_tables(async_session_maker):
    async with async_session_maker() as session:
        await session.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
        await session.commit()
        print("\U0001f9f9 所有表数据已清空")
