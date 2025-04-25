from sqlalchemy import (
    Column, Integer, String, Enum, ForeignKey, Boolean, DateTime, CheckConstraint, LargeBinary, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.mysql import BIGINT
import enum
from sqlalchemy import PrimaryKeyConstraint
from datetime import datetime

Base = declarative_base()

# ========= Tables =========

# ---------------------------
# 模型表
# ---------------------------

class ArchType(enum.Enum):
    CNN = "CNN"
    RNN = "RNN"
    TRANSFORMER = "Transformer"
class Media_type(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
class Trainname(enum.Enum):
    PRETRAIN = "pretrain"
    FINETUNE = "fine-tune"
    RL = "reinforcement learning"

class TaskType(enum.Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    GENERATION = "generation"
    REGRESSION = "regression"

class Model(Base):
    __tablename__ = 'model'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50))
    param_num = Column(BIGINT(unsigned=True))
    media_type = Column(Enum(Media_type), nullable=False)
    arch_name = Column(Enum(ArchType), nullable=False)
    trainname = Column(String(50), nullable=False)
    param = Column(LargeBinary, nullable=False)

    authors = relationship("ModelAuthor", back_populates="model", cascade="all, delete-orphan")
    datasets = relationship("ModelDataset", back_populates="model", cascade="all, delete-orphan")
    tasks = relationship("ModelTask", back_populates="model", cascade="all, delete-orphan")

    cnn = relationship("CNN", uselist=False, back_populates="model")
    rnn = relationship("RNN", uselist=False, back_populates="model")
    transformer = relationship("Transformer", uselist=False, back_populates="model")

class Task_name(enum.Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    GENERATION = "generation"
    SEGMENTATION = "segmentation"
class ModelTask(Base):
    __tablename__ = 'model_tasks'

    model_id = Column(Integer, ForeignKey("model.model_id", ondelete='CASCADE'))
    task_name = Column(Enum(Task_name), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('model_id', 'task_name', name='pk_model_task'),
    )
    model = relationship("Model", back_populates="tasks", passive_deletes=True)


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
    __table_args__ = (
        CheckConstraint('decoder_num >= 0', name='decoder_num'),
        CheckConstraint('attn_size >= 0', name='attn_size'),
        CheckConstraint('up_size >= 0', name='up_size'),
        CheckConstraint('down_size >= 0', name='down_size'),
        CheckConstraint('embed_size >= 0', name='embed_size'),
    )
    model = relationship("Model", back_populates="transformer")


# ---------------------------
# CNN 模型（子表）
# ---------------------------
class CNN(Base):
    __tablename__ = 'cnn'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    module_num = Column(Integer)
    __table_args__ = (
        CheckConstraint('module_num >= 0', name='module_num'),
    )
    model = relationship("Model", back_populates="cnn")
    modules = relationship("Module", back_populates="cnn")

class POOLING_TYPE(enum.Enum):
    MAX = "max"
    MIN = "min"
    AVG = "avg"
    OTHER = "other"

class Module(Base):
    __tablename__ = 'module'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("cnn.model_id", ondelete='CASCADE'), nullable=False)
    conv_size = Column(Integer, nullable=False)
    pool_type = Column(Enum(POOLING_TYPE), nullable=False)

    __table_args__ = (
        CheckConstraint('conv_size >= 0', name='conv_size'),
        # 👇 Ensure (model_id, id) is unique
        UniqueConstraint('model_id', 'id', name='uq_model_id_module_id')
    )

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
    __table_args__ = (
        CheckConstraint('batch_size >= 0', name='batch_size'),
        CheckConstraint('input_size >= 0', name='input_size'),
    )
    model = relationship("Model", back_populates="rnn")


# ---------------------------
# 数据集表（Dataset）
# ---------------------------
class Dataset(Base):
    __tablename__ = 'Dataset'

    ds_id = Column(Integer, primary_key=True, autoincrement=True)
    ds_name = Column(String(255), nullable=False)
    ds_size = Column(Integer, nullable=False)
    media = Column(Enum(Media_type), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    description = Column(String(1000))  # 添加描述字段
    
    __table_args__ = (
        CheckConstraint('ds_size >= 0', name='ds_size'),
    )
    models = relationship("ModelDataset", back_populates="dataset")
    columns = relationship("DsCol", back_populates="dataset", cascade='all, delete-orphan')
    users = relationship("DatasetAuthor", back_populates="dataset")
    Dataset_TASK = relationship("Dataset_TASK", back_populates="Task_relation", cascade="all, delete-orphan")



class Dataset_TASK(Base):
    __tablename__ = 'Dataset_TASK'

    ds_id = Column(Integer, ForeignKey("Dataset.ds_id", ondelete='CASCADE'))
    task = Column(Enum(Task_name), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint('ds_id', 'task', name='pk_dataset_task'),
    )
    Task_relation = relationship("Dataset", back_populates="Dataset_TASK")

class DsCol(Base):
    __tablename__ = 'ds_col'

    ds_id = Column(Integer, ForeignKey("Dataset.ds_id", ondelete='CASCADE'))
    col_name = Column(String(50))
    col_datatype = Column(String(20))
    __table_args__ = (
        PrimaryKeyConstraint('ds_id', 'col_name', 'col_datatype', name='pk_dscol'),
    )

    dataset = relationship("Dataset", back_populates="columns")


# ---------------------------
# 用户表(User)
# ---------------------------
class User(Base):
    __tablename__ = 'user'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String(50), unique = True)
    password_hash = Column(String(100))  # 应使用加密哈希
    affiliate = Column(String(50))
    is_admin = Column(Boolean, default=True)  # 新增管理员字段
    
    models = relationship("ModelAuthor", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    datasets = relationship("DatasetAuthor", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    affiliations = relationship("UserAffil", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

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

    user_id = Column(Integer, ForeignKey("user.user_id", ondelete='CASCADE'), primary_key=True)
    affil_id = Column(Integer, ForeignKey("affil.affil_id", ondelete='CASCADE'), primary_key=True)

    user = relationship("User", back_populates="affiliations", passive_deletes=True)
    affiliation = relationship("Affil", back_populates="users", passive_deletes=True)


# ---------------------------
# 用户-数据集 多对多中间表
# ---------------------------
class DatasetAuthor(Base):
    __tablename__ = 'user_ds'

    user_id = Column(Integer, ForeignKey("user.user_id", ondelete='CASCADE'), primary_key=True)
    ds_id = Column(Integer, ForeignKey("Dataset.ds_id", ondelete='CASCADE'), primary_key=True)

    user = relationship("User", back_populates="datasets", passive_deletes=True)
    dataset = relationship("Dataset", back_populates="users", passive_deletes=True)


# ---------------------------
# 模型-作者 多对多中间表
# ---------------------------
class ModelAuthor(Base):
    __tablename__ = 'model_author'

    model_id = Column(Integer, ForeignKey("model.model_id", ondelete='CASCADE'), primary_key=True)
    user_id = Column(Integer, ForeignKey("user.user_id", ondelete='CASCADE'), primary_key=True)

    model = relationship("Model", back_populates="authors", passive_deletes=True)
    user = relationship("User", back_populates="models", passive_deletes=True)


# ---------------------------
# 模型-数据集 多对多中间表
# ---------------------------
class ModelDataset(Base):
    __tablename__ = 'model_dataset'

    model_id = Column(Integer, ForeignKey("model.model_id", ondelete='CASCADE'), primary_key=True)
    dataset_id = Column(Integer, ForeignKey("Dataset.ds_id", ondelete='CASCADE'), primary_key=True)

    model = relationship("Model", back_populates="datasets", passive_deletes=True)
    dataset = relationship("Dataset", back_populates="models", passive_deletes=True)
