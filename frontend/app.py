# main.py
#streamlit run ./frontend/app.py [ARGUMENTS]
#(replace by your absolute path)
import streamlit as st
from pathlib import Path
import pandas as pd
import sys
import asyncio
from datetime import datetime
import nest_asyncio
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
import frontend.database_api as db_api
from database.database_interface import (
     get_model_by_id, list_datasets, get_dataset_by_id,
    list_users, get_user_by_id, list_affiliations, init_database,
    create_user, update_user, delete_user, get_dataset_info, get_model_info,
    get_model_ids_by_attribute, get_dataset_ids_by_attribute
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
import os
from dotenv import load_dotenv
import hashlib
from typing import List, Dict
from io import BytesIO
from frontend.config import APP_CONFIG
from frontend.db import get_db_session
from frontend.auth import AuthManager
from frontend.components import Sidebar, DatasetUploader, UserManager, ModelUploader, create_search_section


# Allow nested event loops
nest_asyncio.apply()

# Initialize page configuration (must be at the beginning)
st.set_page_config(**APP_CONFIG)

def create_pagination(items, type, page_size=10, page_key="default"):
    """创建分页
    Args:
        items: 数据项列表，可以是对象或字典
        type: 数据类型，如 "models" 或 "datasets"
        page_size: 每页显示的数量
        page_key: 分页状态的唯一标识
    """
    # 获取当前页码
    page_state_key = f'current_page_num_{page_key}'
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 1
        
    # 计算总页数
    total_pages = (len(items) + page_size - 1) // page_size
    
    # 获取当前页的项目
    start_idx = (st.session_state[page_state_key] - 1) * page_size
    end_idx = min(start_idx + page_size, len(items))
    current_items = items[start_idx:end_idx]
    
    # 显示当前页的项目
    for idx, item in enumerate(current_items):
        with st.container(border=True):
            col1, col2 = st.columns([5, 0.7])
            with col1:
                # 处理对象和字典两种情况
                if isinstance(item, dict):
                    name = item.get('ds_name' if type == 'datasets' else 'model_name', '')
                    if type == "models":
                        st.write(f"### {name}")
                        st.caption(f"架构: {item.get('arch_name', '')} | 媒体类型: {item.get('media_type', '')} | 参数量: {item.get('param_num', 0):,}")
                    else:
                        st.write(f"### {name}")
                        tasks = item.get('task', [])
                        st.caption(f"类型: {item.get('media', '')} | 任务: {', '.join(tasks)} | 大小: {item.get('ds_size', 0)/1024:.1f}KB")
                else:
                    st.write(f"### {item.model_name if type=='models' else item.ds_name}")
                    if type == "models":
                        st.caption(f"架构: {item.arch_name.value} | 媒体类型: {item.media_type.value} | 参数量: {item.param_num:,}")
                    else:
                        tasks = [task.task.value for task in item.Dataset_TASK] if hasattr(item, 'Dataset_TASK') else []
                        st.caption(f"类型: {item.media} | 任务: {', '.join(tasks)} | 大小: {item.ds_size/1024:.1f}KB")
            
            with col2:
                item_id = item.get('ds_id' if type == 'datasets' else 'model_id', 0) if isinstance(item, dict) else (item.model_id if type == 'models' else item.ds_id)
                if st.button("查看详情", key=f"{type[:-1]}_{item_id}_{idx}", use_container_width=True):
                    st.session_state[f"selected_{type[:-1]}"] = item
                    st.session_state.current_page = f"{type[:-1]}_detail"
                    st.rerun()
    
    # 分页控制
    _, _, col3 = st.columns([10, 10, 3.5])
    with col3:
        col1, col2, col3 = st.columns([1.8, 1, 1])
        with col1:
            st.button(f'{st.session_state[page_state_key]}/{total_pages}', disabled=True, key=f"page_num_{page_key}")
        with col2:
            if st.button("←", key=f"prev_{page_key}"):
                st.session_state[page_state_key] = max(1, st.session_state[page_state_key] - 1)
                st.rerun()
        with col3:
            if st.button("→", key=f"next_{page_key}"):
                st.session_state[page_state_key] = min(total_pages, st.session_state[page_state_key] + 1)
                st.rerun()

def parse_csv_columns(file_data: bytes) -> List[Dict]:
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

# Async execution decorator
def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# Database session management
@st.cache_resource
def get_db_session():
    load_dotenv()
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", 3306)
    TARGET_DB = os.getenv("TARGET_DB")

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    engine = create_async_engine(
        f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}",
        echo=True
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# User authentication state management
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# File upload handling
def handle_file_upload():
    with st.expander("Upload New Dataset"):
        with st.form("dataset_upload"):
            name = st.text_input("Dataset Name")
            desc = st.text_area("Description")
            file = st.file_uploader("Select Data File", type=["csv", "txt"])
            if st.form_submit_button("Submit"):
                if file:
                    file_path = db_api.db_save_file(file.getvalue(), file.name)
                    db_api.db_create_dataset(name, desc, file_path)
                    st.success("Dataset uploaded successfully!")
                else:
                    st.error("Please select a file")

# File download handling
def handle_file_download(dataset):
    file_data = db_api._file(dataset.ds_name + ".zip")
    if file_data:
        st.download_button(
            label="Download",
            data=file_data,
            file_name=f"{dataset.ds_name}.zip",
            key=f"download_{dataset.ds_id}"
        )
    else:
        st.error("File does not exist")
        
def download_model(model):
    """Handle model download"""
    file_data = db_api.db_get_file(f"{model.model_name}.pt", file_type="models")
    if file_data:
        st.download_button(
            label="Download Model",
            data=file_data,
            file_name=f"{model.model_name}.pt",
            mime="application/octet-stream",
            key=f"download_model_{model.model_id}"
        )
    else:
        st.error("Model file does not exist")

def download_dataset(dataset):
    """Handle dataset download"""
    # Try different possible file extensions
    for ext in ['.txt', '.csv', '.zip']:
        file_data = db_api.db_get_file(f"{dataset.ds_name}{ext}", file_type="datasets")
        if file_data:
            st.download_button(
                label="Download Dataset",
                data=file_data,
                file_name=f"{dataset.ds_name}{ext}",
                mime="text/plain" if ext in ['.txt', '.csv'] else "application/zip",
                key=f"download_dataset_{dataset.ds_id}"
            )
            return True
    
    st.error(f"Dataset file {dataset.ds_name}.* does not exist")
    return False
        
# Login form
def login_form():
    with st.form("Login", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        use_encryption = st.checkbox("Use Encrypted Login", value=True)
        if st.form_submit_button("Login"):
            # Use hashed password for verification (example uses sha256, production should use bcrypt)
            #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
            hashed_pwd = password  # Use plaintext password directly
            if use_encryption:
                user = db_api.db_authenticate_user(username, hashed_pwd)
            else:
                # Non-encrypted login
                user = db_api.db_get_user_by_username(username)
                if user and user.password_hash == password:
                    user = user
                else:
                    user = None

            if user:
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "user_id": user.user_id,
                    "username": user.user_name,
                    "role": "admin" if user.is_admin else "user"
                }
                st.rerun()
            else:
                st.error("Username or password incorrect")

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.title("OpenModelHub")
        if not st.session_state.authenticated:
            login_form()
        else:
            st.success(f"Welcome, {st.session_state.current_user['username']}!")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()
            
        menu_items = ["Home", "Model Repository", "Datasets", "User Management"]
        if st.session_state.current_user and st.session_state.current_user["role"] == "admin":
            menu_items += ["System Management"]
            
        return st.radio("Navigation Menu", menu_items)

# Home page
def render_home():
    """Render home page"""
    st.header("Platform Overview")
    
    # Direct call to database API
    models = db_api.db_list_models()
    datasets = db_api.db_list_datasets()
    users = db_api.db_list_users()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", len(models))
    with col2:
        st.metric("Total Datasets", len(datasets))
    with col3:
        st.metric("Registered Users", len(users))
    with col4:
        st.metric("今日下载量", 2543)
    
    # 添加导出数据按钮
    if st.button("导出所有数据"):
        try:
            # 获取所有数据
            json_data = db_api.db_export_all_data()
            if json_data:
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_{timestamp}.json"
                
                # 转换为JSON字符串，确保使用UTF-8编码
                import json
                json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
                
                # 提供下载
                st.download_button(
                    label="点击下载数据",
                    data=json_str.encode('utf-8'),  # 确保使用UTF-8编码
                    file_name=filename,
                    mime="application/json"
                )
            else:
                st.error("导出数据失败")
        except Exception as e:
            st.error(f"导出数据时出错：{str(e)}")
    
    # 添加注册功能
    if not st.session_state.get('authenticated'):
        st.markdown("---")
        st.subheader("新用户注册")
        with st.form("注册", clear_on_submit=True):
            new_username = st.text_input("用户名")
            new_password = st.text_input("密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            
            if st.form_submit_button("注册"):
                if new_username and new_password:
                    if new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                        return
                    try:
                        # 检查用户名是否已存在
                        existing_user = db_api.db_get_user_by_username(new_username)
                        if existing_user:
                            st.error("用户名已存在")
                            return
                            
                        # 创建新用户，默认非管理员
                        user = db_api.db_create_user(new_username, new_password, is_admin=False)
                        if user:
                            st.success("注册成功！请使用新账号登录")
                        else:
                            st.error("注册失败，请重试")
                    except Exception as e:
                        st.error(f"注册失败：{str(e)}")
                else:
                    st.error("请填写用户名和密码")

def render_datasets():
    """Render dataset management page"""
    st.title("Dataset Management")
    
    # 搜索功能区
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询")
        with col2:
            dataset_attr = st.selectbox(
                "选择字段",
                ["ds_id", "ds_name", "ds_size", "media", "created_at"]
            )
            dataset_val = st.text_input("输入查询值")
        with col3:
            search_clicked = st.button("搜索", key="dataset_search", use_container_width=True)
    
    # 处理搜索逻辑
    if search_clicked:
        if search_query:  # 自然语言搜索
            results, query_info = db_api.db_agent_query(search_query, instance_type=2)  # 2 表示数据集类型
            with st.expander("查询详情"):
                st.json({
                    '自然语言查询': query_info['natural_language_query'],
                    '生成SQL': query_info['generated_sql'],
                    '错误代码': query_info['error_code'],
                    '是否有结果': query_info['has_results'],
                    '错误信息': query_info.get('error', None),
                    '查询结果': results
                })
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                return
        else:  # 属性搜索
            try:
                # 使用 asyncio.run 创建新的事件循环
                ids = asyncio.run(get_dataset_ids_by_attribute(get_db_session()(), dataset_attr, dataset_val))
                if not ids:
                    st.info("未找到符合条件的数据集")
                else:
                    st.session_state.filtered_ids = ids
            except Exception as e:
                st.error(f"搜索失败：{str(e)}")
                return
    
    # 数据集上传
    with st.expander("上传新数据集", expanded=False):
        with st.form("dataset_upload", clear_on_submit=True):
            name = st.text_input("数据集名称*")
            desc = st.text_area("描述")
            media_type = st.selectbox("媒体类型", ["text", "image", "audio", "video"])
            task_type = st.selectbox("任务类型", ["classification", "detection", "generation"])
            file = st.file_uploader("选择数据文件*", type=["txt", "csv"])
            
            if st.form_submit_button("提交"):
                if not name or not file:
                    st.error("带*的字段为必填项")
                else:
                    try:
                        # 保存文件
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # 创建数据集
                        dataset_data = {
                            "ds_name": name,
                            "ds_size": len(file.getvalue()),
                            "media": media_type,
                            "task": [task_type],
                            "columns": [
                                {"col_name": "content", "col_datatype": "text"}
                            ],
                            "description": desc
                        }
                        db_api.db_create_dataset(name, dataset_data)
                        st.success("数据集上传成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
    
    # 获取所有数据集
    datasets = db_api.db_list_datasets()
    
    if not datasets:
        st.info("暂无可用数据集")
        return
    
    # 过滤数据集（如果进行了属性搜索）
    if search_clicked and not search_query and st.session_state.get('filtered_ids'):
        datasets = [d for d in datasets if d.ds_id in st.session_state.get('filtered_ids', [])]
    
    # 显示数据集列表
    create_pagination(datasets, "datasets")
    
    # 显示数据集详情
    if st.session_state.get("current_page") == "dataset_detail":
        dataset = st.session_state.get("selected_dataset")
        if dataset:
            st.markdown("---")
            st.subheader(f"数据集详情 - {dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '')}")
            
            # 显示描述
            st.write("**描述：**")
            st.write(dataset.description if hasattr(dataset, 'description') else dataset.get('description', '无描述'))
            
            # 显示任务信息
            st.write("**任务类型：**")
            if hasattr(dataset, 'Dataset_TASK'):
                tasks = [task.task.value for task in dataset.Dataset_TASK]
            else:
                tasks = dataset.get('task', [])
            st.write(", ".join(tasks) if tasks else "无任务")
            
            # 显示数据集大小
            st.write("**数据集大小：**")
            size = dataset.ds_size if hasattr(dataset, 'ds_size') else dataset.get('ds_size', 0)
            st.write(f"{size/1024:.1f}KB")
            
            # 下载按钮
            if st.button("下载数据集", key=f"download_{dataset.ds_id if hasattr(dataset, 'ds_id') else dataset.get('ds_id', '')}"):
                file_data = db_api.db_get_file(dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '') + ".txt")
                if file_data:
                    st.download_button(
                        label="点击下载",
                        data=file_data,
                        file_name=f"{dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("文件不存在")
            
            # 返回按钮
            if st.button("返回列表", key="back_to_list"):
                st.session_state.current_page = "datasets"
                st.rerun()

# Model repository
def render_models():
    """Render model repository page"""
    st.title("Model Repository")
    
    # 搜索功能区
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")
        with col2:
            model_attr = st.selectbox(
                "选择字段",
                ["model_id", "model_name", "param_num", "media_type", "arch_name", "trainname"]
            )
            model_val = st.text_input("输入查询值")
        with col3:
            search_clicked = st.button("搜索", key="model_search", use_container_width=True)
    
    # 处理搜索逻辑
    if search_clicked:
        if search_query:  # 自然语言搜索
            results, query_info = db_api.db_agent_query(search_query, instance_type=1)  # 1 表示模型类型
            with st.expander("查询详情"):
                st.json({
                    '自然语言查询': query_info['natural_language_query'],
                    '生成SQL': query_info['generated_sql'],
                    '错误代码': query_info['error_code'],
                    '是否有结果': query_info['has_results'],
                    '错误信息': query_info.get('error', None),
                    '查询结果': results
                })
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                return
        else:  # 属性搜索
            try:
                # 使用 asyncio.run 创建新的事件循环
                ids = asyncio.run(get_model_ids_by_attribute(get_db_session()(), model_attr, model_val))
                if not ids:
                    st.info("未找到符合条件的模型")
                else:
                    st.session_state.filtered_ids = ids
            except Exception as e:
                st.error(f"搜索失败：{str(e)}")
                return
    
    # 模型上传
    with st.expander("上传新模型", expanded=False):
        with st.form("model_upload", clear_on_submit=True):
            name = st.text_input("模型名称*")
            param_num = st.number_input("参数量", min_value=1000, value=1000000)
            
            col1, col2 = st.columns(2)
            with col1:
                arch_type = st.selectbox(
                    "架构类型*",
                    ["CNN", "RNN", "TRANSFORMER"],
                    help="选择模型架构类型"
                )
                media_type = st.selectbox(
                    "媒体类型*",
                    ["TEXT", "IMAGE", "AUDIO", "VIDEO"],
                    help="选择适用媒体类型"
                )
            with col2:
                train_type = st.selectbox(
                    "训练类型*",
                    ["PRETRAIN", "FINETUNE", "RL"],
                    help="选择训练类型"
                )
                selected_tasks = st.multiselect(
                    "任务类型*",
                    ["CLASSIFICATION", "DETECTION", "GENERATION", "SEGMENTATION"],
                    default=["CLASSIFICATION"]
                )
            
            file = st.file_uploader("选择模型文件*", type=["pt", "pth", "ckpt"])
            
            if st.form_submit_button("提交"):
                if not name or not file:
                    st.error("带*的字段为必填项")
                else:
                    try:
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        model_data = {
                            "model_name": name,
                            "param_num": param_num,
                            "arch_name": arch_type,
                            "media_type": media_type,
                            "tasks": selected_tasks,
                            "trainname": train_type,
                            "param": str(file_path)
                        }
                        db_api.db_create_model(model_data)
                        st.success("模型上传成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
    
    # 获取所有模型
    models = db_api.db_list_models()
    
    if not models:
        st.info("暂无可用模型")
        return
    
    # 过滤模型（如果进行了属性搜索）
    if search_clicked and not search_query and st.session_state.get('filtered_ids'):
        models = [m for m in models if m.model_id in st.session_state.get('filtered_ids', [])]
    
    # 显示模型列表
    create_pagination(models, "models")
    
    # 显示模型详情
    if st.session_state.get("current_page") == "model_detail":
        model = st.session_state.get("selected_model")
        if model:
            st.markdown("---")
            st.subheader(f"模型详情 - {model.model_name if hasattr(model, 'model_name') else model.get('model_name', '')}")
            
            # 显示基本信息
            st.write("**基本信息：**")
            info_data = {
                "模型ID": model.model_id if hasattr(model, 'model_id') else model.get('model_id', ''),
                "架构类型": model.arch_name.value if hasattr(model, 'arch_name') else model.get('arch_name', ''),
                "参数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else f"{model.get('param_num', 0):,}",
                "训练类型": model.trainname.value if hasattr(model, 'trainname') else model.get('trainname', '')
            }
            st.table(pd.DataFrame(list(info_data.items()), columns=["属性", "值"]))
            
            # 显示关联信息
            st.write("**关联信息：**")
            if hasattr(model, 'tasks'):
                tasks = [task.task_name.value for task in model.tasks]
            else:
                tasks = model.get('tasks', [])
            
            if hasattr(model, 'authors'):
                authors = [author.user_name for author in model.authors]
            else:
                authors = model.get('authors', [])
            
            if hasattr(model, 'datasets'):
                datasets = [dataset.ds_name for dataset in model.datasets]
            else:
                datasets = model.get('datasets', [])
            
            rel_data = {
                "支持任务": ", ".join(tasks),
                "作者": ", ".join(authors),
                "关联数据集": ", ".join(datasets)
            }
            st.table(pd.DataFrame(list(rel_data.items()), columns=["类型", "名称"]))
            
            # 下载按钮
            if st.button("下载模型", key=f"download_{model.model_id if hasattr(model, 'model_id') else model.get('model_id', '')}"):
                file_data = db_api.db_get_file(model.model_name if hasattr(model, 'model_name') else model.get('model_name', '') + ".pt")
                if file_data:
                    st.download_button(
                        label="点击下载",
                        data=file_data,
                        file_name=f"{model.model_name if hasattr(model, 'model_name') else model.get('model_name', '')}.pt",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("文件不存在")
            
            # 返回按钮
            if st.button("返回列表", key="back_to_list"):
                st.session_state.current_page = "models"
                st.rerun()

def get_current_page_data(page_type: str):
    """获取当前页面的数据
    Args:
        page_type: 页面类型，可以是 "models", "datasets", "users" 等
    Returns:
        当前页面的数据列表
    """
    try:
        if page_type == "models":
            return db_api.db_list_models()
        elif page_type == "datasets":
            return db_api.db_list_datasets()
        elif page_type == "users":
            return db_api.db_list_users()
        else:
            return []
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return []

# User management (admin function)
def render_users():
    """Render user management page"""
    user_manager = UserManager()
    user_manager.render()

# Default login function
def default_login():
    """Use default account to login, for development testing"""
    if not st.session_state.get('authenticated'):
        st.session_state.authenticated = True
        st.session_state.current_user = {
            "user_id": 1,
            "username": "admin",
            "role": "admin"
        }

def main():
    """Main program entry"""
    # Development mode: use default login
    # default_login()  # Uncomment to enable default login
    
    # Normal mode: use authentication manager
    auth_manager = AuthManager()
    sidebar = Sidebar(auth_manager)
    
    # Get current page
    page = sidebar.render()
    
    # 主页不需要登录权限
    if page == "主页":
        render_home()
        return
        
    # 其他页面需要登录权限
    if not auth_manager.is_authenticated():
        st.warning("请先登录以访问此页面")
        return
    
    # 路由到对应页面
    if page == "模型仓库":
        render_models()
    elif page == "数据集":
        render_datasets()
    elif page == "用户管理" and auth_manager.is_admin():
        render_users()
    elif page == "系统管理":
        st.write("系统管理功能正在开发中...")

if __name__ == "__main__":
    try:
        # Start application directly
        main()
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        print(f"Error details: {str(e)}")
