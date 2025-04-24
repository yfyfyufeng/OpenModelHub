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

def render_datasets():
    """渲染数据集仓库页面"""
    st.title("数据集仓库")

    uploader = DatasetUploader()
    if uploader.render():
        st.rerun()

    # 添加搜索输入框
    search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询")

    # 添加搜索按钮
    if st.button("搜索", key="dataset_search"):
        if search_query:
            results, query_info = db_api.db_agent_query(search_query)
            with st.expander("查询详情"):
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("无查询结果")
            return

    # 获取所有数据集
    datasets = db_api.db_list_datasets()

    # 创建事件循环用于异步获取详细信息
    if 'dataset_loop' not in st.session_state:
        st.session_state.dataset_loop = asyncio.new_event_loop()
    loop = st.session_state.dataset_loop

    # 预先获取所有数据集的详细信息
    dataset_details = {}
    for dataset in datasets:
        async def get_details(ds_id):
            async with get_db_session()() as session:
                return await get_dataset_info(session, ds_id)

        details = loop.run_until_complete(get_details(dataset.ds_id))
        if details:
            dataset_details[dataset.ds_id] = details

    # 展示数据集列表
    dataset_data = []
    for dataset in datasets:
        details = dataset_details.get(dataset.ds_id, {})

        columns = []
        if details and "columns" in details:
            columns = [f"{col.get('col_name', '')}/{col.get('col_datatype', '')}" for col in details.get("columns", [])]
        columns_str = ", ".join(columns) if columns else ""

        tasks = []
        if details and "tasks" in details:
            for task in details.get("tasks", []):
                if hasattr(task, 'value'):
                    tasks.append(str(task.value))
                else:
                    tasks.append(str(task))
        tasks_str = ", ".join(tasks) if tasks else ""

        models = []
        if details and "models" in details:
            models = [str(model) for model in details.get("models", [])]
        models_str = ", ".join(models) if models else ""

        authors = []
        if details and "authors" in details:
            authors = [str(author) for author in details.get("authors", [])]
        authors_str = ", ".join(authors) if authors else ""

        dataset_data.append({
            "ID": dataset.ds_id,
            "名称": dataset.ds_name,
            "大小": f"{dataset.ds_size:,}" if hasattr(dataset, 'ds_size') else "",
            "媒体类型": str(dataset.media.value) if hasattr(dataset.media, 'value') else str(dataset.media),
            "创建时间": dataset.created_at.strftime("%Y-%m-%d") if dataset.created_at else "",
            "数据集列（列名/数据类型）": columns_str[:50] + ("..." if len(columns_str) > 50 else ""),
            "支持任务": tasks_str,
            "关联模型": models_str[:50] + ("..." if len(models_str) > 50 else ""),
            "作者": authors_str
        })

    df = pd.DataFrame(dataset_data)

    st.markdown("""
    <style>
    .stDataFrame table {
        font-size: 16px !important;
    }
    .stDataFrame td, .stDataFrame th {
        font-size: 16px !important;
        padding: 8px !important;
        text-align: center !important;  /* 文字居中 */
        vertical-align: middle !important; /* 垂直居中 */
    }
    .stDataFrame th {
        font-size: 17px !important;
        font-weight: bold !important;
        text-align: center !important;  /* 表头文字居中 */
    }
    [data-testid="stDataFrameContainer"] {
        width: 100%;
        overflow-x: auto !important;
    }
    .stTable {
        width: 100%;
        max-height: none !important;
    }
    /* 确保内容与单元格匹配 */
    .stDataFrame td div, .stDataFrame th div {
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("数据集ID", width="small"),
            "名称": st.column_config.TextColumn("数据集名称", width="medium"),
            "大小": st.column_config.TextColumn("数据量", width="small"),
            "媒体类型": st.column_config.TextColumn("媒体类型", width="small"),
            "创建时间": st.column_config.TextColumn("创建时间", width="small"),
            "数据集列": st.column_config.TextColumn("数据集列", width="medium"),
            "支持任务": st.column_config.TextColumn("支持任务", width="medium"),
            "关联模型": st.column_config.TextColumn("关联模型", width="medium"),
            "作者": st.column_config.TextColumn("作者", width="medium")
        },
        hide_index=True,
        use_container_width=True
    )

    # 数据集详情查看部分
    selected_id = st.number_input("输入数据集ID查看详情", min_value=1, step=1)
    if selected_id:
        loop = st.session_state.dataset_loop

        async def get_dataset_details():
            async with get_db_session()() as session:
                return await get_dataset_info(session, selected_id)

        dataset_info = loop.run_until_complete(get_dataset_details())

        if dataset_info:
            with st.expander(f"数据集详情 – {dataset_info['dataset']['ds_name']} --basic", expanded=False):
                # Basic info as dataframe
                st.subheader("基本信息")
                basic_info = {
                    "数据集名称": dataset_info['dataset']['ds_name'],
                    "数据集ID": dataset_info['dataset']['ds_id'],
                    "数据量": f"{dataset_info['dataset']['ds_size']:,}",
                    "媒体类型": str(dataset_info['dataset']['media'].value)
                    if hasattr(dataset_info['dataset']['media'], 'value')
                    else str(dataset_info['dataset']['media']),
                    "创建时间": dataset_info['dataset']['created_at'].strftime("%Y-%m-%d %H:%M")
                    if dataset_info['dataset']['created_at'] else ""
                }
                df_basic = pd.DataFrame(list(basic_info.items()), columns=["属性", "值"])
                st.dataframe(df_basic, hide_index=True, use_container_width=True)

            with st.expander(f"数据集详情 – {dataset_info['dataset']['ds_name']} --detailed", expanded=False):

                st.subheader("数据集列")
                if dataset_info['columns']:
                    df_cols = pd.DataFrame({
                        "列名/数据类型": [
                            f"{c.get('col_name', '')}/{c.get('col_datatype', '')}"
                            for c in dataset_info['columns']
                        ]
                    })
                    st.dataframe(df_cols, hide_index=True, use_container_width=True)
                else:
                    st.info("无列信息")

                st.subheader("支持任务")
                if dataset_info['tasks']:
                    df_tasks = pd.DataFrame({
                        "任务": [
                            str(t.value) if hasattr(t, 'value') else str(t)
                            for t in dataset_info['tasks']
                        ]
                    })
                    st.dataframe(df_tasks, hide_index=True, use_container_width=True)
                else:
                    st.info("无任务信息")

                st.subheader("关联模型")
                if dataset_info['models']:
                    df_models = pd.DataFrame({"模型": [str(m) for m in dataset_info['models']]})
                    st.dataframe(df_models, hide_index=True, use_container_width=True)
                else:
                    st.info("无关联模型")

                st.subheader("数据集作者")
                if dataset_info['authors']:
                    df_authors = pd.DataFrame({"作者": [str(a) for a in dataset_info['authors']]})
                    st.dataframe(df_authors, hide_index=True, use_container_width=True)
                else:
                    st.info("无作者信息")
        else:
            st.error(f"未找到ID为{selected_id}的数据集")

def render_models():
    """渲染模型仓库页面"""
    st.title("模型仓库")

    # 搜索输入和按钮
    search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")
    if st.button("搜索", key="model_search"):
        if search_query:
            results, query_info = db_api.db_agent_query(search_query)
            with st.expander("查询详情"):
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    return

    # 创建事件循环用于异步获取详细信息
    if 'model_loop' not in st.session_state:
        st.session_state.model_loop = asyncio.new_event_loop()
    loop = st.session_state.model_loop

    # 加载所有模型
    models = db_api.db_list_models()

    # 预先获取所有模型的详细信息
    model_details = {}
    for model in models:
        async def get_details(model_id):
            async with get_db_session()() as session:
                return await get_model_info(session, model_id)

        details = loop.run_until_complete(get_details(model.model_id))
        if details:
            model_details[model.model_id] = details

    def safe_get_value(obj, attr_name):
        if isinstance(obj, dict):
            val = obj.get(attr_name, None)
        else:
            val = getattr(obj, attr_name, None)
        from enum import Enum
        if isinstance(val, Enum):
            return val.name
        if isinstance(val, str) and '.' in val:
            return val.split('.')[-1]
        return str(val)

    # 列表展示
    model_data = []
    for model in models:
        details = model_details.get(model.model_id, {})

        # 任务信息
        tasks = []
        if hasattr(model, 'tasks') and model.tasks:
            for t in model.tasks:
                from enum import Enum
                if isinstance(t.task_name, Enum):
                    tasks.append(str(t.task_name.value))
                else:
                    tasks.append(str(t.task_name))
        elif details and 'tasks' in details:
            tasks = [str(task) for task in details['tasks']]

        tasks_str = ", ".join(tasks) if tasks else "未知"

        # 获取作者信息 - 从详情中提取
        authors = []
        if details and 'authors' in details:
            authors = [str(author) for author in details['authors']]
        authors_str = ", ".join(authors) if authors else "未知"

        # 获取关联数据集信息 - 从详情中提取
        datasets = []
        if details and 'datasets' in details:
            datasets = [str(dataset) for dataset in details['datasets']]
        datasets_str = ", ".join(datasets) if datasets else "未知"

        # 安全取值并捕获异常
        try:
            arch = safe_get_value(model, 'arch_name')
            media = safe_get_value(model, 'media_type')
            train = safe_get_value(model, 'trainname')
        except Exception as e:
            st.error(f"获取模型属性失败: {e}")
            continue

        model_data.append({
            "ID": model.model_id,
            "名称": model.model_name,
            "架构": arch,
            "媒体类型": media,
            "参数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else "未知",
            "训练名称": train,
            "任务": tasks_str,
            "作者": authors_str[:50] + ("..." if len(authors_str) > 50 else ""),
            "关联数据集": datasets_str[:50] + ("..." if len(datasets_str) > 50 else "")
        })

    # 创建并显示数据框
    df = pd.DataFrame(model_data)

    # 应用样式
    st.markdown("""
    <style>
    .stDataFrame table {
        font-size: 16px !important;
    }
    .stDataFrame td, .stDataFrame th {
        font-size: 16px !important;
        padding: 8px !important;
        text-align: center !important;  /* 文字居中 */
        vertical-align: middle !important; /* 垂直居中 */
    }
    .stDataFrame th {
        font-size: 17px !important;
        font-weight: bold !important;
        text-align: center !important;  /* 表头文字居中 */
    }
    [data-testid="stDataFrameContainer"] {
        width: 100%;
        overflow-x: auto !important;
    }
    .stTable {
        width: 100%;
        max-height: none !important;
    }
    /* 确保内容与单元格匹配 */
    .stDataFrame td div, .stDataFrame th div {
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("模型ID", width="small"),
            "名称": st.column_config.TextColumn("模型名称", width="medium"),
            "架构": st.column_config.TextColumn("架构类型", width="small"),
            "媒体类型": st.column_config.TextColumn("适用媒体", width="small"),
            "参数量": st.column_config.TextColumn("参数量", width="small"),
            "训练名称": st.column_config.TextColumn("训练方式", width="small"),
            "任务": st.column_config.TextColumn("支持任务", width="medium"),
            "作者": st.column_config.TextColumn("作者", width="medium"),
            "关联数据集": st.column_config.TextColumn("关联数据集", width="medium")
        },
        hide_index=True,
        use_container_width=True
    )

    # 详情查看部分
    selected_id = st.number_input("输入模型ID查看详情", min_value=1, step=1)
    if selected_id:
        loop = st.session_state.model_loop

        async def get_model_details():
            async with get_db_session()() as session:
                return await get_model_info(session, selected_id)

        model_info = loop.run_until_complete(get_model_details())

        if model_info:
            with st.expander(f"模型详情 – {model_info['model_name']} --basic", expanded=False):
                # Basic info as dataframe
                st.subheader("基本信息")
                basic_info = {
                    "模型ID": model_info['model_id'],
                    "架构类型": safe_get_value(model_info, 'arch_name'),
                    "参数量": f"{model_info['param_num']:,}",
                    "媒体类型": safe_get_value(model_info, 'media_type'),
                    "训练方式": safe_get_value(model_info, 'trainname')
                }
                df_basic = pd.DataFrame(list(basic_info.items()), columns=["属性", "值"])
                st.dataframe(df_basic, hide_index=True, use_container_width=True)

            with st.expander(f"模型详情 – {model_info['model_name']} --detailed", expanded=False):

                st.subheader("支持任务")
                if model_info.get('tasks'):
                    df_tasks = pd.DataFrame({"任务": [str(t) for t in model_info['tasks']]})
                    st.dataframe(df_tasks, hide_index=True, use_container_width=True)
                else:
                    st.info("无任务信息")

                st.subheader("模型作者")
                if model_info.get('authors'):
                    df_auth = pd.DataFrame({"作者": [str(a) for a in model_info['authors']]})
                    st.dataframe(df_auth, hide_index=True, use_container_width=True)
                else:
                    st.info("无作者信息")

                st.subheader("关联数据集")
                if model_info.get('datasets'):
                    df_ds = pd.DataFrame({"数据集": [str(d) for d in model_info['datasets']]})
                    st.dataframe(df_ds, hide_index=True, use_container_width=True)
                else:
                    st.info("无关联数据集")

                st.subheader("架构详情")
                if model_info.get('cnn'):
                    st.metric("模块数量", model_info['cnn']['module_num'])
                    df_cnn = pd.DataFrame([
                        {"卷积大小": m["conv_size"], "池化类型": m["pool_type"]}
                        for m in model_info['cnn']['modules']
                    ])
                    st.dataframe(df_cnn, hide_index=True, use_container_width=True)

                elif model_info.get('rnn'):
                    df_rnn = pd.DataFrame({
                        "批量大小": [model_info['rnn']['batch_size']],
                        "输入大小": [model_info['rnn']['input_size']],
                        "准则": [model_info['rnn']['criteria']]
                    })
                    st.dataframe(df_rnn, hide_index=True, use_container_width=True)

                elif model_info.get('transformer'):
                    tf = model_info['transformer']
                    df_tf = pd.DataFrame({
                        "解码器数量": [tf['decoder_num']],
                        "注意力大小": [tf['attn_size']],
                        "上升尺寸": [tf['up_size']],
                        "下降尺寸": [tf['down_size']],
                        "嵌入尺寸": [tf['embed_size']],
                    })
                    st.dataframe(df_tf, hide_index=True, use_container_width=True)
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