from sqlalchemy import (
    Column, Integer, String, Enum, ForeignKey
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.mysql import BIGINT
import enum
from sqlalchemy import PrimaryKeyConstraint

Base = declarative_base()

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

    authors = relationship("ModelAuthor", back_populates="model", cascade="all, delete-orphan")
    datasets = relationship("ModelDataset", back_populates="model", cascade="all, delete-orphan")
    tasks = relationship("ModelTask", back_populates="model", cascade="all, delete-orphan")

    cnn = relationship("CNN", uselist=False, back_populates="model")
    rnn = relationship("RNN", uselist=False, back_populates="model")
    transformer = relationship("Transformer", uselist=False, back_populates="model")


class ModelTask(Base):
    __tablename__ = 'model_tasks'

    model_id = Column(Integer, ForeignKey("model.model_id", ondelete='CASCADE'))
    task_name = Column(String(50))
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

    model = relationship("Model", back_populates="transformer")


# ---------------------------
# CNN 模型（子表）
# ---------------------------
class CNN(Base):
    __tablename__ = 'cnn'

    model_id = Column(Integer, ForeignKey("model.model_id"), primary_key=True)
    module_num = Column(Integer)

    model = relationship("Model", back_populates="cnn")
    modules = relationship("Module", back_populates="cnn")


class Module(Base):
    __tablename__ = 'module'

    model_id = Column(Integer, ForeignKey("cnn.model_id", ondelete='CASCADE'))
    conv_size = Column(Integer)
    pool_type = Column(String(20))
    __table_args__ = (
        PrimaryKeyConstraint('model_id', 'conv_size', name='pk_module'),
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
    columns = relationship("DsCol", back_populates="dataset", cascade='all, delete-orphan')
    users = relationship("UserDataset", back_populates="dataset")


class DsCol(Base):
    __tablename__ = 'ds_col'

    ds_id = Column(Integer, ForeignKey("dataset.ds_id", ondelete='CASCADE'))
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
    user_name = Column(String(50))
    affiliate = Column(String(50))

    models = relationship("ModelAuthor", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    datasets = relationship("UserDataset", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
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
class UserDataset(Base):
    __tablename__ = 'user_ds'

    user_id = Column(Integer, ForeignKey("user.user_id", ondelete='CASCADE'), primary_key=True)
    ds_id = Column(Integer, ForeignKey("dataset.ds_id", ondelete='CASCADE'), primary_key=True)

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
    dataset_id = Column(Integer, ForeignKey("dataset.ds_id", ondelete='CASCADE'), primary_key=True)

    model = relationship("Model", back_populates="datasets", passive_deletes=True)
    dataset = relationship("Dataset", back_populates="models", passive_deletes=True)
