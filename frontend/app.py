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
    list_models, get_model_by_id, list_datasets, get_dataset_by_id,
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
from frontend.components import Sidebar, DatasetUploader, UserManager

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
        
# 登录表单
def login_form():
    with st.form("登录", clear_on_submit=True):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.form_submit_button("登录"):
            # 使用哈希后的密码进行验证（示例使用sha256，生产环境应使用bcrypt）
      #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
            hashed_pwd = password  # 直接使用明文密码
            user = db_api.db_authenticate_user(username, hashed_pwd)
            if user  :
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
    
    # 添加搜索输入框
    search_query = st.text_input("搜索模型", placeholder="输入自然语言查询，例如：'查找所有准确率大于90%的模型'")
    
    # 添加搜索按钮
    if st.button("搜索", key="model_search"):
        if search_query:
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
    
    # 如果没有搜索或搜索无结果，显示所有模型
    models = db_api.db_list_models()
    
    # 展示模型列表
    df = pd.DataFrame([{
        "ID": model.model_id,
        "名称": model.model_name,
        "类型": model.arch_name.value,
        "参数数量": f"{model.param_num:,}"
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
    
    # 添加搜索输入框
    search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询，例如：'查找所有图像分类数据集'")
    
    # 添加搜索按钮
    if st.button("搜索", key="dataset_search"):
        if search_query:
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
    uploader = DatasetUploader()
    if uploader.render():
        st.rerun()
    
    # 如果没有搜索或搜索无结果，显示所有数据集
    datasets = db_api.db_list_datasets()
    
    if not datasets:
        st.info("暂无数据集")
        return
    
    # 显示数据集信息
    for dataset in datasets:
        with st.container(border=True):
            st.subheader(dataset.ds_name)
            # 获取数据集的任务
            tasks = [task.task.value for task in dataset.Dataset_TASK]  # 获取枚举值
            task_str = ", ".join(tasks) if tasks else "无任务"
            st.caption(f"类型：{dataset.media} | 任务：{task_str} | 大小：{dataset.ds_size/1024:.1f}KB")
            
            if st.button("查看详情", key=f"dataset_{dataset.ds_id}"):
                st.session_state.selected_dataset = dataset
                st.session_state.current_page = "dataset_detail"

# 用户管理（管理员功能）
def render_users():
    """渲染用户管理页面"""
    user_manager = UserManager()
    user_manager.render()

# 主程序逻辑
def main():
    """主程序入口"""
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