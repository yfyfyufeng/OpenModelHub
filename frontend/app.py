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
    """渲染模型仓库页面（附带 DEBUG 信息）"""
    st.title("模型仓库")
    st.write("[DEBUG] Entered render_models()")

    # 搜索输入和按钮
    search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")
    st.write(f"[DEBUG] search_query = {search_query!r}")
    if st.button("搜索", key="model_search"):
        st.write("[DEBUG] Search button clicked")
        if search_query:
            results, query_info = db_api.db_agent_query(search_query)
            st.write(f"[DEBUG] db_agent_query returned {len(results)} rows, query_info={query_info!r}")
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

    # 加载所有模型
    models = db_api.db_list_models()
    st.write(f"[DEBUG] Loaded {len(models)} models from db_list_models()")

    # 定义安全取值函数
    def safe_get_value(obj, attr_name):
        st.write(f"[DEBUG][safe_get] obj={obj!r} type={type(obj)}, attr_name={attr_name!r}")
        if isinstance(obj, dict):
            val = obj.get(attr_name, None)
        else:
            val = getattr(obj, attr_name, None)
        st.write(f"[DEBUG][safe_get] raw val={val!r} type={type(val)}")
        from enum import Enum
        if isinstance(val, Enum):
            st.write(f"[DEBUG][safe_get] using .value on {val!r}")
            return str(val.value)
        return str(val) if val is not None else "未知"

    # 列表展示
    model_data = []
    for idx, model in enumerate(models):
        st.write(f"[DEBUG][LIST] idx={idx} id={model.model_id}, "
                 f"arch_name={model.arch_name!r}({type(model.arch_name)}), "
                 f"media_type={model.media_type!r}({type(model.media_type)}), "
                 f"trainname={model.trainname!r}({type(model.trainname)})")

        # 任务信息
        tasks = []
        if hasattr(model, 'tasks') and model.tasks:
            for t in model.tasks:
                st.write(f"[DEBUG][LIST][TASK] model_id={model.model_id}, "
                         f"t.task_name={t.task_name!r}({type(t.task_name)})")
                from enum import Enum
                if isinstance(t.task_name, Enum):
                    tasks.append(str(t.task_name.value))
                else:
                    tasks.append(str(t.task_name))
        task_str = ", ".join(tasks) if tasks else "未知"

        # 安全取值并捕获异常
        try:
            arch   = safe_get_value(model, 'arch_name')
            media  = safe_get_value(model, 'media_type')
            train  = safe_get_value(model, 'trainname')
        except Exception as e:
            st.error(f"[ERROR][LIST] model_id={model.model_id} safe_get_value failed: {e}")
            raise

        model_data.append({
            "ID": model.model_id,
            "名称": model.model_name,
            "架构": arch,
            "媒体类型": media,
            "参数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else "未知",
            "训练名称": train,
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

    # 详情查看
    selected_id = st.number_input("输入模型ID查看详情", min_value=1, step=1)
    st.write(f"[DEBUG] selected_id = {selected_id}")
    if selected_id:
        if 'model_loop' not in st.session_state:
            st.session_state.model_loop = asyncio.new_event_loop()
            st.write("[DEBUG] Created new asyncio event loop")
        loop = st.session_state.model_loop

        async def get_model_details():
            async with get_db_session()() as session:
                return await get_model_info(session, selected_id)

        model_info = loop.run_until_complete(get_model_details())
        st.write(f"[DEBUG][DETAIL] fetched model_info = {model_info!r}")

        if not model_info:
            st.error(f"未找到ID为{selected_id}的模型")
            return

        # 详情展开
        with st.expander(f"模型详情 - {model_info['model_name']}", expanded=True):
            # 再次捕获安全取值
            try:
                arch_name  = safe_get_value(model_info, 'arch_name')
                media_type = safe_get_value(model_info, 'media_type')
                trainname  = safe_get_value(model_info, 'trainname')
            except Exception as e:
                st.error(f"[ERROR][DETAIL] safe_get_value failed: {e}")
                raise

            # 基本信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("模型ID", model_info['model_id'])
                st.metric("架构类型", arch_name)
            with col2:
                st.metric("参数量", f"{model_info['param_num']:,}")
                st.metric("媒体类型", media_type)
            with col3:
                st.metric("训练方式", trainname)

            # 支持任务
            st.subheader("支持任务")
            if model_info.get('tasks'):
                st.write(f"[DEBUG][DETAIL] tasks raw = {model_info['tasks']!r}")
                tasks = []
                from enum import Enum
                for t in model_info['tasks']:
                    if isinstance(t, Enum):
                        tasks.append(str(t.value))
                    else:
                        tasks.append(str(t))
                st.dataframe(pd.DataFrame({"任务": tasks}),
                             hide_index=True, use_container_width=True)
            else:
                st.info("无任务信息")

            # 模型作者
            st.subheader("模型作者")
            if model_info.get('authors'):
                st.write(f"[DEBUG][DETAIL] authors raw = {model_info['authors']!r}")
                authors = [str(a) for a in model_info['authors']]
                st.dataframe(pd.DataFrame({"作者": authors}),
                             hide_index=True, use_container_width=True)
            else:
                st.info("无作者信息")

            # 关联数据集
            st.subheader("关联数据集")
            if model_info.get('datasets'):
                st.write(f"[DEBUG][DETAIL] datasets raw = {model_info['datasets']!r}")
                datasets = [str(d) for d in model_info['datasets']]
                st.dataframe(pd.DataFrame({"数据集": datasets}),
                             hide_index=True, use_container_width=True)
            else:
                st.info("无关联数据集")

            # 架构详情
            st.subheader("架构详情")
            st.write(f"[DEBUG][DETAIL] cnn section: {model_info.get('cnn')!r}, rnn: {model_info.get('rnn')!r}, transformer: {model_info.get('transformer')!r}")
            # CNN
            if model_info.get('cnn'):
                st.write(f"[DEBUG][DETAIL] Rendering CNN modules: {model_info['cnn'].get('modules')!r}")
                st.write(f"模块数量: {model_info['cnn']['module_num']}")
                if model_info['cnn'].get('modules'):
                    dfm = pd.DataFrame([
                        {"模块": f"模块 {i+1}",
                         "卷积尺寸": m['conv_size'],
                         "池化类型": m['pool_type']}
                        for i, m in enumerate(model_info['cnn']['modules'])
                    ])
                    st.dataframe(dfm, hide_index=True, use_container_width=True)
            # RNN
            elif model_info.get('rnn'):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("评价标准", str(model_info['rnn']['criteria']))
                with c2:
                    st.metric("批次大小", model_info['rnn']['batch_size'])
                with c3:
                    st.metric("输入尺寸", model_info['rnn']['input_size'])
            # Transformer
            elif model_info.get('transformer'):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("解码器数量", model_info['transformer']['decoder_num'])
                    st.metric("注意力尺寸", model_info['transformer']['attn_size'])
                with c2:
                    st.metric("上采样尺寸", model_info['transformer']['up_size'])
                    st.metric("下采样尺寸", model_info['transformer']['down_size'])
                with c3:
                    st.metric("嵌入尺寸", model_info['transformer']['embed_size'])
            else:
                st.info("无架构详情")

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