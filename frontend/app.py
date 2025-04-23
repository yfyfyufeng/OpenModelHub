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
    create_user, update_user, delete_user, get_dataset_info, get_model_info
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

# 默认登录admin用户
if not st.session_state.authenticated:
    user = db_api.db_authenticate_user("admin", "admin")
    if user:
        st.session_state.authenticated = True
        st.session_state.current_user = {
            "user_id": user.user_id,
            "username": user.user_name,
            "role": "admin" if user.is_admin else "user"
        }
        st.rerun()

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

# # 模型仓库
# def render_models():
#     """渲染模型仓库页面"""
#     st.title("模型仓库")
#
#     # 添加搜索输入框
#     search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")
#
#     # 添加搜索按钮
#     if st.button("搜索", key="model_search"):
#         if search_query:
#             results, query_info = db_api.db_agent_query(search_query)
#             # 显示查询详情
#             with st.expander("查询详情"):
#                 st.json({
#                     'natural_language_query': query_info['natural_language_query'],
#                     'generated_sql': query_info['generated_sql'],
#                     'error_code': query_info['error_code'],
#                     'has_results': query_info['has_results'],
#                     'error': query_info.get('error', None),
#                     'sql_res': results
#                 })
#             if results:
#                 df = pd.DataFrame(results)
#                 st.dataframe(df)
#                 return
#
#     # 如果没有搜索或搜索无结果，显示所有模型
#     models = db_api.db_list_models()
#
#     # 展示模型列表
#     df = pd.DataFrame([{
#         "ID": model.model_id,
#         "名称": model.model_name,
#         "类型": model.arch_name.value,
#         "参数数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else "未知"
#     } for model in models])
#
#     st.dataframe(
#         df,
#         column_config={
#             "ID": "模型ID",
#             "名称": "模型名称",
#             "类型": "架构类型",
#             "参数数量": "参数量"
#         },
#         hide_index=True,
#         use_container_width=True
#     )
#
#     # 模型详情侧边栏
#     selected_id = st.number_input("输入模型ID查看详情", min_value=1)
#     if selected_id:
#         model = db_api.db_get_model(selected_id)
#         if model:
#             with st.expander(f"模型详情 - {model.model_name}"):
#                 st.write(f"**架构类型**: {model.arch_name.value}")
#                 st.write(f"**适用媒体类型**: {model.media_type}")
#
#                 if model.tasks:
#                     st.write("**支持任务**:")
#                     for task in model.tasks:
#                         st.code(task.task_name)
#
#                 # 下载按钮
#                 if st.button("下载模型"):
#                     st.success("下载开始...（演示用）")
#
# # 修改后的数据集管理
# def render_datasets():
#     """渲染数据集管理页面"""
#     st.title("数据集管理")
#
#     # 添加搜索输入框
#     search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询")
#
#     # 添加搜索按钮
#     if st.button("搜索", key="dataset_search"):
#         if search_query:
#             results, query_info = db_api.db_agent_query(search_query)
#             # 显示查询详情
#             with st.expander("查询详情"):
#                 st.json({
#                     'natural_language_query': query_info['natural_language_query'],
#                     'generated_sql': query_info['generated_sql'],
#                     'error_code': query_info['error_code'],
#                     'has_results': query_info['has_results'],
#                     'error': query_info.get('error', None),
#                     'sql_res': results
#                 })
#             if results:
#                 df = pd.DataFrame(results)
#                 st.dataframe(df)
#                 return
#
#     # 数据集上传
#     with st.expander("上传新数据集"):
#         with st.form("dataset_upload"):
#             name = st.text_input("数据集名称")
#             desc = st.text_area("描述")
#             file = st.file_uploader("选择数据文件", type=["txt"])
#
#             # 任务选择
#             st.write("选择任务类型：")
#             # 预定义的任务类型
#             predefined_tasks = ["classification", "detection", "generation", "segmentation"]
#             selected_tasks = st.multiselect(
#                 "选择任务类型",
#                 predefined_tasks,
#                 default=["classification"],
#                 help="可以选择多个任务类型"
#             )
#
#             if st.form_submit_button("提交"):
#                 if file:
#                     try:
#                         # 保存文件
#                         file_path = db_api.db_save_file(file.getvalue(), file.name)
#
#                         # 创建数据集
#                         dataset_data = {
#                             "ds_name": name,
#                             "ds_size": len(file.getvalue()),
#                             "media": "text",  # 默认类型
#                             "task": selected_tasks,  # 使用选择的任务
#                             "columns": [
#                                 {"col_name": "content", "col_datatype": "text"}
#                             ],
#                             "description": desc  # 添加描述字段
#                         }
#                         db_api.db_create_dataset(name, dataset_data)
#                         st.success("数据集上传成功！")
#                         st.rerun()  # 刷新页面以显示新数据集
#                     except Exception as e:
#                         st.error(f"上传失败：{str(e)}")
#                 else:
#                     st.error("请选择文件")
#
#     # 如果没有搜索或搜索无结果，显示所有数据集
#     datasets = db_api.db_list_datasets()
#
#     if not datasets:
#         st.info("暂无数据集")
#         return
#
#     # 显示数据集信息（按创建时间倒序排列）
#     for dataset in sorted(datasets, key=lambda x: x.created_at, reverse=True):
#         with st.container(border=True):
#             st.subheader(dataset.ds_name)
#             # 获取数据集的任务
#             tasks = [task.task.value for task in dataset.Dataset_TASK]  # 获取枚举值
#             task_str = ", ".join(tasks) if tasks else "无任务"
#             st.caption(f"类型：{dataset.media} | 任务：{task_str} | 大小：{dataset.ds_size/1024:.1f}KB")
#
#             if st.button("查看详情", key=f"dataset_{dataset.ds_id}"):
#                 st.session_state.selected_dataset = dataset
#                 st.session_state.current_page = "dataset_detail"
#
#     # 显示数据集详情
#     if st.session_state.get("current_page") == "dataset_detail":
#         dataset = st.session_state.get("selected_dataset")
#         if dataset:
#             st.markdown("---")
#             st.subheader(f"数据集详情 - {dataset.ds_name}")
#
#             # 显示描述
#             st.write("**描述：**")
#             st.write(dataset.description if hasattr(dataset, 'description') else "暂无描述")
#
#             # 显示任务信息
#             st.write("**任务类型：**")
#             tasks = [task.task.value for task in dataset.Dataset_TASK]
#             st.write(", ".join(tasks) if tasks else "无任务")
#
#             # 显示数据集大小
#             st.write("**数据集大小：**")
#             st.write(f"{dataset.ds_size/1024:.1f}KB")
#
#             # 下载按钮
#             if st.button("下载数据集", key=f"download_{dataset.ds_id}"):
#                 file_data = db_api.db_get_file(dataset.ds_name + ".txt")
#                 if file_data:
#                     st.download_button(
#                         label="点击下载",
#                         data=file_data,
#                         file_name=f"{dataset.ds_name}.txt",
#                         mime="text/plain"
#                     )
#                 else:
#                     st.error("文件不存在")
#
#             # 返回按钮
#             if st.button("返回列表", key="back_to_list"):
#                 st.session_state.current_page = "datasets"
#                 st.rerun()

def render_datasets():
    """渲染数据集仓库页面"""
    st.title("数据集仓库")

    # 添加搜索输入框
    search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询")

    # 添加搜索按钮
    if st.button("搜索", key="dataset_search"):
        if search_query:
            # 搜索逻辑
            pass

    # 如果没有搜索或搜索无结果，显示所有数据集
    datasets = db_api.db_list_datasets()

    # 展示数据集列表
    dataset_data = []
    for dataset in datasets:
        # Convert enum values to strings
        dataset_data.append({
            "ID": dataset.ds_id,
            "名称": dataset.ds_name,
            "大小": f"{dataset.ds_size:,}",
            "媒体类型": str(dataset.media.value) if hasattr(dataset.media, 'value') else str(dataset.media),
            "创建时间": dataset.created_at.strftime("%Y-%m-%d") if dataset.created_at else "",
        })

    df = pd.DataFrame(dataset_data)
    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("数据集ID"),
            "名称": st.column_config.TextColumn("数据集名称"),
            "大小": st.column_config.TextColumn("数据量"),
            "媒体类型": st.column_config.TextColumn("媒体类型"),
            "创建时间": st.column_config.TextColumn("创建时间")
        },
        hide_index=True,
        use_container_width=True
    )

    # 数据集详情查看
    selected_id = st.number_input("输入数据集ID查看详情", min_value=1, step=1)
    if selected_id:
        # 创建一个新的事件循环
        if 'dataset_loop' not in st.session_state:
            st.session_state.dataset_loop = asyncio.new_event_loop()

        # 使用session_state中保存的事件循环
        loop = st.session_state.dataset_loop

        # 定义异步函数
        async def get_dataset_details():
            async with get_db_session()() as session:
                return await get_dataset_info(session, selected_id)

        # 在同一个事件循环中执行异步操作
        dataset_info = loop.run_until_complete(get_dataset_details())

        if dataset_info:
            with st.expander(f"数据集详情 - {dataset_info['dataset']['ds_name']}", expanded=True):
                # 基本信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("数据集ID", dataset_info['dataset']['ds_id'])
                    st.metric("数据量", f"{dataset_info['dataset']['ds_size']:,}")
                with col2:
                    # Convert enum to string
                    media_type = str(dataset_info['dataset']['media'].value) if hasattr(dataset_info['dataset']['media'], 'value') else str(dataset_info['dataset']['media'])
                    st.metric("媒体类型", media_type)
                    st.metric("创建时间", dataset_info['dataset']['created_at'].strftime("%Y-%m-%d %H:%M") if dataset_info['dataset']['created_at'] else "")

                # 列信息
                st.subheader("数据集列")
                if dataset_info['columns']:
                    columns_df = pd.DataFrame(dataset_info['columns'])
                    st.dataframe(columns_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无列信息")

                # 任务信息
                st.subheader("支持任务")
                if dataset_info['tasks']:
                    # Convert task objects to strings if needed
                    tasks = [str(task.value) if hasattr(task, 'value') else str(task) for task in dataset_info['tasks']]
                    tasks_df = pd.DataFrame({"任务": tasks})
                    st.dataframe(tasks_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无任务信息")

                # 关联模型
                st.subheader("关联模型")
                if dataset_info['models']:
                    models_df = pd.DataFrame({"模型": dataset_info['models']})
                    st.dataframe(models_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无关联模型")

                # 作者信息
                st.subheader("数据集作者")
                if dataset_info['authors']:
                    authors_df = pd.DataFrame({"作者": dataset_info['authors']})
                    st.dataframe(authors_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无作者信息")
        else:
            st.error(f"未找到ID为{selected_id}的数据集")

def render_models():
    """渲染模型仓库页面"""
    st.title("模型仓库")

    # 添加搜索输入框
    search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")

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
    model_data = []
    for model in models:
        # 获取任务信息
        tasks = [str(task.task_name) for task in model.tasks] if hasattr(model, 'tasks') and model.tasks else []
        task_str = ", ".join(tasks) if tasks else "未知"

        # Safely extract attributes with proper type checking
        def safe_get_value(obj, attr_name):
            if hasattr(obj, attr_name):
                attr = getattr(obj, attr_name)
                if hasattr(attr, 'value'):  # Check if it's an enum
                    return str(attr.value)
                else:
                    return str(attr)
            return "未知"

        model_data.append({
            "ID": model.model_id,
            "名称": model.model_name,
            "架构": safe_get_value(model, 'arch_name'),
            "媒体类型": safe_get_value(model, 'media_type'),
            "参数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else "未知",
            "训练名称": safe_get_value(model, 'trainname'),
            "任务": task_str
        })

    df = pd.DataFrame(model_data)
    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("模型ID"),
            "名称": st.column_config.TextColumn("模型名称"),
            "架构": st.column_config.TextColumn("架构类型"),
            "媒体类型": st.column_config.TextColumn("适用媒体"),
            "参数量": st.column_config.TextColumn("参数量"),
            "训练名称": st.column_config.TextColumn("训练方式"),
            "任务": st.column_config.TextColumn("支持任务")
        },
        hide_index=True,
        use_container_width=True
    )

    # 模型详情查看
    selected_id = st.number_input("输入模型ID查看详情", min_value=1, step=1)
    if selected_id:
        # 创建一个新的事件循环
        if 'model_loop' not in st.session_state:
            st.session_state.model_loop = asyncio.new_event_loop()

        # 使用session_state中保存的事件循环
        loop = st.session_state.model_loop

        # 定义异步函数
        async def get_model_details():
            async with get_db_session()() as session:
                return await get_model_info(session, selected_id)

        # 在同一个事件循环中执行异步操作
        model_info = loop.run_until_complete(get_model_details())

        if model_info:
            with st.expander(f"模型详情 - {model_info['model_name']}", expanded=True):
                # 基本信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("模型ID", model_info['model_id'])
                    # Safely get architecture name
                    arch_name = safe_get_value(model_info, 'arch_name')
                    st.metric("架构类型", arch_name)
                with col2:
                    st.metric("参数量", f"{model_info['param_num']:,}" if 'param_num' in model_info else "未知")
                    # Safely get media type
                    media_type = safe_get_value(model_info, 'media_type')
                    st.metric("媒体类型", media_type)
                with col3:
                    # Safely get training name
                    trainname = safe_get_value(model_info, 'trainname')
                    st.metric("训练方式", trainname)

                # 任务信息
                st.subheader("支持任务")
                if 'tasks' in model_info and model_info['tasks']:
                    # Convert task objects to strings safely
                    tasks = []
                    for task in model_info['tasks']:
                        if hasattr(task, 'value'):
                            tasks.append(str(task.value))
                        else:
                            tasks.append(str(task))
                    tasks_df = pd.DataFrame({"任务": tasks})
                    st.dataframe(tasks_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无任务信息")

                # 作者信息
                st.subheader("模型作者")
                if 'authors' in model_info and model_info['authors']:
                    # Convert author objects to strings safely
                    authors = []
                    for author in model_info['authors']:
                        if hasattr(author, 'user_name'):
                            authors.append(str(author.user_name))
                        else:
                            authors.append(str(author))
                    authors_df = pd.DataFrame({"作者": authors})
                    st.dataframe(authors_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无作者信息")

                # 关联数据集
                st.subheader("关联数据集")
                if 'datasets' in model_info and model_info['datasets']:
                    # Convert dataset objects to strings safely
                    datasets = []
                    for dataset in model_info['datasets']:
                        if hasattr(dataset, 'ds_name'):
                            datasets.append(str(dataset.ds_name))
                        else:
                            datasets.append(str(dataset))
                    datasets_df = pd.DataFrame({"数据集": datasets})
                    st.dataframe(datasets_df, hide_index=True, use_container_width=True)
                else:
                    st.info("无关联数据集")

                # 架构细节
                st.subheader("架构详情")
                # Use string comparison on the safe value
                arch_name_str = arch_name.upper()
                if "CNN" in arch_name_str and 'cnn' in model_info and model_info['cnn']:
                    st.write(f"模块数量: {model_info['cnn']['module_num']}")

                    if 'modules' in model_info['cnn'] and model_info['cnn']['modules']:
                        modules_data = []
                        for i, module in enumerate(model_info['cnn']['modules']):
                            modules_data.append({
                                "模块": f"模块 {i+1}",
                                "卷积尺寸": module['conv_size'],
                                "池化类型": module['pool_type']
                            })
                        st.dataframe(pd.DataFrame(modules_data), hide_index=True, use_container_width=True)

                elif "RNN" in arch_name_str and 'rnn' in model_info and model_info['rnn']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("评价标准", str(model_info['rnn']['criteria']))
                    with col2:
                        st.metric("批次大小", model_info['rnn']['batch_size'])
                    with col3:
                        st.metric("输入尺寸", model_info['rnn']['input_size'])

                elif "TRANSFORMER" in arch_name_str and 'transformer' in model_info and model_info['transformer']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("解码器数量", model_info['transformer']['decoder_num'])
                        st.metric("注意力尺寸", model_info['transformer']['attn_size'])
                    with col2:
                        st.metric("上采样尺寸", model_info['transformer']['up_size'])
                        st.metric("下采样尺寸", model_info['transformer']['down_size'])
                    with col3:
                        st.metric("嵌入尺寸", model_info['transformer']['embed_size'])
                else:
                    st.info("无架构详情")
        else:
            st.error(f"未找到ID为{selected_id}的模型")

# 用户管理（管理员功能）
def render_users():
    """渲染用户管理页面"""
    user_manager = UserManager()
    user_manager.render()

# 主程序逻辑
def main():
    """主程序入口"""
    # 暂时注释掉认证相关代码
    # auth_manager = AuthManager()
    # sidebar = Sidebar(auth_manager)

    # 获取当前页面
    # page = sidebar.render()

    # 检查认证状态
    # if not auth_manager.is_authenticated() and page != "主页":
    #     st.warning("请先登录以访问该页面")
    #     return

    # 路由到对应页面
    # if page == "主页":
    #     render_home()
    # elif page == "模型仓库":
    #     render_models()
    # elif page == "数据集":
    #     render_datasets()
    # elif page == "用户管理" and auth_manager.is_admin():
    #     render_users()
    # elif page == "系统管理":
    #     st.write("系统管理功能开发中...")

    # 简化版本：直接显示所有页面

    st.sidebar.title("OpenModelHub")
    page = st.sidebar.radio("导航菜单", ["主页", "模型仓库", "数据集"])

    if page == "主页":
        render_home()
    elif page == "模型仓库":
        render_models()
    elif page == "数据集":
        render_datasets()

if __name__ == "__main__":
    try:
        # 直接启动应用
        main()
    except Exception as e:
        st.error(f"应用启动失败：{str(e)}")
        print(f"错误详情：{str(e)}")