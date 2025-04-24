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
    create_user, update_user, delete_user
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


# 允许嵌套事件循环
nest_asyncio.apply()

# 初始化页面配置（必须在最前面）
st.set_page_config(**APP_CONFIG)

def parse_csv_columns(file_data: bytes) -> List[Dict]:
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

# 异步执行装饰器
def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# 数据库会话管理
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

# 用户认证状态管理
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# 文件上传处理
def handle_file_upload():
    with st.expander("上传新数据集"):
        with st.form("dataset_upload"):
            name = st.text_input("数据集名称")
            desc = st.text_area("描述")
            file = st.file_uploader("选择数据文件", type=["csv", "txt"])
            if st.form_submit_button("提交"):
                if file:
                    file_path = db_api.db_save_file(file.getvalue(), file.name)
                    db_api.db_create_dataset(name, desc, file_path)
                    st.success("数据集上传成功！")
                else:
                    st.error("请选择文件")

# 文件下载处理
def handle_file_download(dataset):
    file_data = db_api._file(dataset.ds_name + ".zip")
    if file_data:
        st.download_button(
            label="下载",
            data=file_data,
            file_name=f"{dataset.ds_name}.zip",
            key=f"download_{dataset.ds_id}"
        )
    else:
        st.error("文件不存在")
        
def download_model(model):
    """处理模型下载"""
    file_data = db_api.db_get_file(f"{model.model_name}.pt", file_type="models")
    if file_data:
        st.download_button(
            label="下载模型",
            data=file_data,
            file_name=f"{model.model_name}.pt",
            mime="application/octet-stream",
            key=f"download_model_{model.model_id}"
        )
    else:
        st.error("模型文件不存在")

def download_dataset(dataset):
    """处理数据集下载"""
    # 尝试不同的可能文件扩展名
    for ext in ['.txt', '.csv', '.zip']:
        file_data = db_api.db_get_file(f"{dataset.ds_name}{ext}", file_type="datasets")
        if file_data:
            st.download_button(
                label="下载数据集",
                data=file_data,
                file_name=f"{dataset.ds_name}{ext}",
                mime="text/plain" if ext in ['.txt', '.csv'] else "application/zip",
                key=f"download_dataset_{dataset.ds_id}"
            )
            return True
    
    st.error(f"数据集文件 {dataset.ds_name}.* 不存在")
    return False
        
# 登录表单
def login_form():
    with st.form("登录", clear_on_submit=True):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        use_encryption = st.checkbox("使用加密登录", value=True)
        if st.form_submit_button("登录"):
            # 使用哈希后的密码进行验证（示例使用sha256，生产环境应使用bcrypt）
            #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
            hashed_pwd = password  # 直接使用明文密码
            if use_encryption:
                user = db_api.db_authenticate_user(username, hashed_pwd)
            else:
                # 非加密登录
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
                st.error("用户名或密码错误")

# 侧边栏导航
def sidebar():
    with st.sidebar:
        st.title("OpenModelHub")
        if not st.session_state.authenticated:
            login_form()
        else:
            st.success(f"欢迎，{st.session_state.current_user['username']}！")
            if st.button("退出登录"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()
            
        menu_items = ["主页", "模型仓库", "数据集", "用户管理"]
        if st.session_state.current_user and st.session_state.current_user["role"] == "admin":
            menu_items += ["系统管理"]
            
        return st.radio("导航菜单", menu_items)

# 主页
def render_home():
    """渲染主页"""
    st.header("平台概览")
    
    # 直接调用数据库API
    models = db_api.db_list_models()
    datasets = db_api.db_list_datasets()
    users = db_api.db_list_users()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总模型数", len(models))
    with col2:
        st.metric("总数据集", len(datasets))
    with col3:
        st.metric("注册用户", len(users))
    with col4:
        st.metric("今日下载量", 2543)

# 模型仓库
def render_models():
    """渲染模型仓库页面"""
    st.title("模型仓库")
    
     # 使用统一的搜索部分
    if create_search_section("models"):
        return
    
    # 添加搜索输入框
    st.markdown("""
        <style>
        .stButton > button {
            margin-top: 25px;  /* Adjust this value to match your input height */
        }
        div.row-widget.stSelectbox {
            margin-top: 25px;  /* Match the button margin */
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([4.7, 0.3])
        
        with col1:
            search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")
        with col2:
            search_clicked = st.button("搜索", key="model_search", use_container_width=True)
    
    if search_query and search_clicked:
        results, query_info = db_api.db_agent_query(search_query)
        # 显示查询详情
        with st.expander("查询详情"):
            st.json({
                'natural_language_query': query_info['natural_language_query'],
                'generated_sql': query_info['generated_sql'],
                'error_code': query_info['error_code'],
                'has_results': query_info['has_results'],
                'error': query_info.get('error', None),
                'sql_res': results
            })
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            return
    
    # 模型上传
    with st.expander("上传新模型"):
        with st.form("model_upload"):
            name = st.text_input("模型名称")
            param_num = st.number_input("参数量", min_value=1000, value=1000000)
            
            # 架构类型选择
            arch_types = ["CNN", "RNN", "TRANSFORMER"]
            arch_type = st.selectbox(
                "架构类型",
                arch_types,
                help="选择模型架构类型"
            )
            
            # 媒体类型选择
            media_types = ["TEXT", "IMAGE", "AUDIO", "VIDEO"]
            media_type = st.selectbox(
                "媒体类型",
                media_types,
                help="选择模型适用的媒体类型"
            )
            
            # 任务选择
            task_types = ["CLASSIFICATION", "DETECTION", "GENERATION", "SEGMENTATION"]
            selected_tasks = st.multiselect(
                "任务类型",
                task_types,
                default=["CLASSIFICATION"],
                help="可以选择多个任务类型"
            )
            
            # 训练类型选择
            train_types = ["PRETRAIN", "FINETUNE", "RL"]
            train_type = st.selectbox(
                "训练类型",
                train_types,
                help="选择模型的训练类型"
            )
            
            # 文件上传
            file = st.file_uploader("选择模型文件", type=["pt", "pth", "ckpt", "bin", "txt"])
            
            if st.form_submit_button("提交"):
                if file and name:
                    try:
                        # 保存文件
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # 创建模型
                        model_data = {
                            "model_name": name,
                            "param_num": param_num,
                            "arch_name": arch_type,
                            "media_type": media_type,
                            "tasks": selected_tasks,
                            "trainname": train_type,
                            "param": file_path
                        }
                        db_api.db_create_model(model_data)
                        st.success("模型上传成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
                else:
                    st.error("请填写模型名称并选择文件")
    
    # 显示所有模型
    models = db_api.db_list_models()
    
    if not models:
        st.info("暂无模型")
        return
    
    # 显示模型信息（按创建时间倒序排列）
    for model in sorted(models, key=lambda x: x.created_at if hasattr(x, 'created_at') else datetime.now(), reverse=True):
        with st.container(border=True):
            st.subheader(model.model_name)
            # 获取模型的任务
            tasks = [task.task_name.value for task in model.tasks] if hasattr(model, 'tasks') else []
            task_str = ", ".join(tasks) if tasks else "无任务"
            st.caption(f"架构：{model.arch_name.value} | 媒体类型：{model.media_type.value} | 参数量：{model.param_num:,}")
            
            if st.button("查看详情", key=f"model_{model.model_id}"):
                st.session_state.selected_model = model
                st.session_state.current_page = "model_detail"
    
    # 显示模型详情
    if st.session_state.get("current_page") == "model_detail":
        model = st.session_state.get("selected_model")
        if model:
            st.markdown("---")
            st.subheader(f"模型详情 - {model.model_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**架构类型：**")
                st.write(model.arch_name.value)
                
                st.write("**媒体类型：**")
                st.write(model.media_type.value)
                
                st.write("**参数量：**")
                st.write(f"{model.param_num:,}")
            
            with col2:
                st.write("**训练类型：**")
                st.write(model.trainname.value)
                
                st.write("**支持任务：**")
                tasks = [task.task_name.value for task in model.tasks] if hasattr(model, 'tasks') else []
                st.write(", ".join(tasks) if tasks else "无任务")
            
            # 下载按钮
            if st.button("下载模型", key=f"download_{model.model_id}"):
                file_data = db_api.db_get_file(f"{model.model_name}.pt")
                if file_data:
                    st.download_button(
                        label="点击下载",
                        data=file_data,
                        file_name=f"{model.model_name}.pt",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("文件不存在")
            
            # 返回按钮
            if st.button("返回列表", key="back_to_list"):
                st.session_state.current_page = "models"
                st.rerun()
    
    # 如果没有搜索或搜索无结果，显示所有模型
    models = db_api.db_list_models()
    
    # 展示模型列表
    df = pd.DataFrame([{
        "ID": model.model_id,
        "名称": model.model_name,
        "类型": model.arch_name.value,
        "参数数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else "未知"
    } for model in models])
    
    st.dataframe(
        df,
        column_config={
            "ID": "模型ID",
            "名称": "模型名称",
            "类型": "架构类型",
            "参数数量": "参数量"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 模型详情侧边栏
    selected_id = st.number_input("输入模型ID查看详情", min_value=1)
    if selected_id:
        model = db_api.db_get_model(selected_id)
        if model:
            with st.expander(f"模型详情 - {model.model_name}"):
                st.write(f"**架构类型**: {model.arch_name.value}")
                st.write(f"**适用媒体类型**: {model.media_type}")
                
                if model.tasks:
                    st.write("**支持任务**:")
                    for task in model.tasks:
                        st.code(task.task_name)
                
                # 下载按钮
                if st.button("下载模型"):
                    st.success("下载开始...（演示用）")

# 修改后的数据集管理
def render_datasets():
    """渲染数据集管理页面"""
    st.title("数据集管理")
    
    # 使用统一的搜索部分
    if create_search_section("datasets"):
        return
    
    # 添加搜索输入框
    st.markdown("""
        <style>
        .stButton > button {
            margin-top: 25px;  /* Adjust this value to match your input height */
        }
        div.row-widget.stSelectbox {
            margin-top: 25px;  /* Match the button margin */
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([4.7, 0.3])
        
        with col1:
            search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询")
        with col2:
            search_clicked = st.button("搜索", key="dataset_search", use_container_width=True)
    if search_query and search_clicked:
        results, query_info = db_api.db_agent_query(search_query)
        # 显示查询详情
        with st.expander("查询详情"):
            st.json({
                'natural_language_query': query_info['natural_language_query'],
                'generated_sql': query_info['generated_sql'],
                'error_code': query_info['error_code'],
                'has_results': query_info['has_results'],
                'error': query_info.get('error', None),
                'sql_res': results
            })
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            return
    
    # 数据集上传
    with st.expander("上传新数据集"):
        with st.form("dataset_upload"):
            name = st.text_input("数据集名称")
            desc = st.text_area("描述")
            file = st.file_uploader("选择数据文件", type=["txt"])
            
            # 任务选择
            st.write("选择任务类型：")
            # 预定义的任务类型
            predefined_tasks = ["classification", "detection", "generation", "segmentation"]
            selected_tasks = st.multiselect(
                "选择任务类型",
                predefined_tasks,
                default=["classification"],
                help="可以选择多个任务类型"
            )
            
            if st.form_submit_button("提交"):
                if file:
                    try:
                        # 保存文件
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # 创建数据集
                        dataset_data = {
                            "ds_name": name,
                            "ds_size": len(file.getvalue()),
                            "media": "text",  # 默认类型
                            "task": selected_tasks,  # 使用选择的任务
                            "columns": [
                                {"col_name": "content", "col_datatype": "text"}
                            ],
                            "description": desc,  # 添加描述字段
                            "created_at": datetime.now()  # 添加创建时间
                        }
                        db_api.db_create_dataset(name, dataset_data)
                        st.success("数据集上传成功！")
                        st.rerun()  # 刷新页面以显示新数据集
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
                else:
                    st.error("请选择文件")
    
    # 如果没有搜索或搜索无结果，显示所有数据集
    datasets = db_api.db_list_datasets()
    
    if not datasets:
        st.info("暂无数据集")
        return
    
    # 显示数据集信息（按创建时间倒序排列）
    for dataset in sorted(datasets, key=lambda x: x.created_at, reverse=True):
        with st.container(border=True):
            st.subheader(dataset.ds_name)
            # 获取数据集的任务
            tasks = [task.task.value for task in dataset.Dataset_TASK]  # 获取枚举值
            task_str = ", ".join(tasks) if tasks else "无任务"
            st.caption(f"类型：{dataset.media} | 任务：{task_str} | 大小：{dataset.ds_size/1024:.1f}KB")
            
            if st.button("查看详情", key=f"dataset_{dataset.ds_id}"):
                st.session_state.selected_dataset = dataset
                st.session_state.current_page = "dataset_detail"
    
    # 显示数据集详情
    if st.session_state.get("current_page") == "dataset_detail":
        dataset = st.session_state.get("selected_dataset")
        if dataset:
            st.markdown("---")
            st.subheader(f"数据集详情 - {dataset.ds_name}")
            
            # 显示描述
            st.write("**描述：**")
            st.write(dataset.description if hasattr(dataset, 'description') else "暂无描述")
            
            # 显示任务信息
            st.write("**任务类型：**")
            tasks = [task.task.value for task in dataset.Dataset_TASK]
            st.write(", ".join(tasks) if tasks else "无任务")
            
            # 显示数据集大小
            st.write("**数据集大小：**")
            st.write(f"{dataset.ds_size/1024:.1f}KB")
            
            # 下载按钮
            if st.button("下载数据集", key=f"download_{dataset.ds_id}"):
                file_data = db_api.db_get_file(dataset.ds_name + ".txt")
                if file_data:
                    st.download_button(
                        label="点击下载",
                        data=file_data,
                        file_name=f"{dataset.ds_name}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("文件不存在")
            
            # 返回按钮
            if st.button("返回列表", key="back_to_list"):
                st.session_state.current_page = "datasets"
                st.rerun()

# 用户管理（管理员功能）
def render_users():
    """渲染用户管理页面"""
    user_manager = UserManager()
    user_manager.render()

# 默认登录函数
def default_login():
    """使用默认账号登录，用于开发测试"""
    if not st.session_state.get('authenticated'):
        st.session_state.authenticated = True
        st.session_state.current_user = {
            "user_id": 1,
            "username": "admin",
            "role": "admin"
        }

def main():
    """主程序入口"""
    # 开发模式：使用默认登录
    # default_login()  # 取消注释以启用默认登录
    
    # 正常模式：使用认证管理器
    auth_manager = AuthManager()
    sidebar = Sidebar(auth_manager)
    
    # 获取当前页面
    page = sidebar.render()
    
    # 检查认证状态
    if not auth_manager.is_authenticated() and page != "主页":
        st.warning("请先登录以访问该页面")
        return
    
    # 路由到对应页面
    if page == "主页":
        render_home()
    elif page == "模型仓库":
        render_models()
    elif page == "数据集":
        render_datasets()
    elif page == "用户管理" and auth_manager.is_admin():
        render_users()
    elif page == "系统管理":
        st.write("系统管理功能开发中...")

if __name__ == "__main__":
    try:
        # 直接启动应用
        main()
    except Exception as e:
        st.error(f"应用启动失败：{str(e)}")
        print(f"错误详情：{str(e)}")