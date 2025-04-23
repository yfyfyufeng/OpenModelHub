import os
import re
from dotenv import load_dotenv

load_dotenv()

# ----------------------
# Auxiliary Functions
# ----------------------

# language handling

def clean_sql_output(output: str) -> str:
    # 去除开头的 ```sql 或 sql
    output = re.sub(r"^\s*```sql\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(r"^\s*sql\s*", "", output, flags=re.IGNORECASE)
    # 去除末尾的 ```
    output = re.sub(r"\s*```$", "", output)
    return output.strip()

# ----------------------
# 🌐 Proxy Settings (done before importing asyncio)
# -----------------------

env_proxy = [
    "http_proxy", "https_proxy", "ftp_proxy", "all_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY"
]

original_env = {}
for i in env_proxy:
    # ----- Step 1: Backup original proxy settings -----
    original_env[i] = os.environ.get(i)
    # ----- Step 2: Temporarily clear proxy -----
    os.environ.pop(i, None)

# ----------------------
# 📦 Import Librarie
# -----------------------

import asyncio
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import text

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    # proxies= None
)

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")

DATABASE_URL = f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

# ----------------------
# 📘 Prompt
# ----------------------
SYSTEM_PROMPT = """
You are an SQL generator. Please generate MySQL queries based on natural language requests.
You must only generate standard SQL (ANSI-compatible) queries.
Do NOT use MySQL-specific syntax like DESCRIBE or SHOW.
If the user asks to see the structure of a table, use the information_schema.columns table.

The database schema is as follows:

- model(model_id, model_name, param_num, media_type, arch_name)
  - model_name: Name of the model
  - param_num: Number of parameters
  - media_type: Media type used by the model (e.g., 'text', 'image', 'audio', 'video')
  - arch_name: Architecture of the model (e.g., 'CNN', 'RNN', 'Transformer')

- model_tasks(model_id, task_name)
  - task_name: Task associated with the model (e.g., 'classification', 'detection', 'generation')

- cnn(model_id, module_num)
  - module_num: Number of modules in the CNN model

- module(model_id, conv_size, pool_type)
  - conv_size: Size of the convolution layer
  - pool_type: Type of pooling used in the model (e.g., 'max', 'avg')

- transformer(model_id, decoder_num, attn_size, up_size, down_size, embed_size)
  - decoder_num: Number of decoders in the transformer model
  - attn_size: Attention size
  - up_size: Size of the upward transformation
  - down_size: Size of the downward transformation
  - embed_size: Embedding size

- rnn(model_id, criteria, batch_size, input_size)
  - criteria: Loss function criteria used in the RNN model
  - batch_size: Batch size used in training the model
  - input_size: Size of the input data for the model

- dataset(ds_id, ds_name, ds_size, media, task, created_at)
  - ds_name: Name of the dataset
  - ds_size: Size of the dataset
  - media: Type of media in the dataset (e.g., 'text', 'image')
  - task: Tasks associated with the dataset (e.g., 'classification', 'detection')
  - created_at: Creation timestamp of the dataset

- ds_col(ds_id, col_name, col_datatype)
  - col_name: Column name in the dataset
  - col_datatype: Data type of the column

- user(user_id, user_name, password_hash, affiliate, is_admin)
  - user_name: Name of the user
  - password_hash: Hashed password for the user
  - affiliate: The organization to which the user is affiliated
  - is_admin: Boolean flag indicating if the user is an admin

- affil(affil_id, affil_name)
  - affil_name: Name of the affiliation (organization)

- user_affil(user_id, affil_id)
  - Maps users to affiliations

- model_author(model_id, user_id)
  - Maps models to their authors

- model_dataset(model_id, dataset_id)
  - Maps models to datasets

- user_ds(user_id, ds_id)
  - Maps users to datasets

# Synonym Handling:
- If the user query refers to a "language model", but no such table or attribute exists,
you should map "language model" to "models where media_type = 'text'".
- If the user refers to concepts like "text data models", translate this to models where `media_type = 'text'`.

# Domain Constraints:
1. Ensure that all queries respect the database schema and constraints such as:
    - Task types for models and datasets must be chosen from predefined options (e.g., 'classification', 'detection').
    - Models' architecture types must be one of 'CNN', 'RNN', or 'Transformer'.
    - Media types for models and datasets are limited to 'text', 'image', 'audio', 'video'.
2. You should never reference columns or tables that do not exist in the schema.


Only return the SQL query. Do not add explanations.
"""


# ----------------------
# 🔁 Generate SQL with GPT
# ----------------------
async def natural_language_to_sql(nl_input: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_input}
        ],
        temperature=0,
    )
    return clean_sql_output(response.choices[0].message.content)

# ----------------------
# ❌ Error Fixing
# ----------------------
async def fix_sql_with_error(nl_input: str, original_sql: str, error_msg: str) -> str:
    fix_prompt = f"""
The original natural language request is:
{nl_input}

The generated SQL is:
{original_sql}

The following error occurred during execution:
{error_msg}

Please fix this SQL query and return a correct SQL query.
"""
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fix_prompt}
        ],
        temperature=0,
    )
    return clean_sql_output(response.choices[0].message.content)

# ----------------------
# 📦 Execute SQL
# ----------------------
async def execute_sql(sql: str, session=None):
    """执行SQL查询，可以使用外部提供的会话"""
    if session:
        # 使用提供的会话
        try:
            result = await session.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows], None
        except Exception as e:
            return None, str(e)
    else:
        # 使用自己的会话（用于独立运行时）
        async with SessionLocal() as session:
            try:
                result = await session.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows], None
            except Exception as e:
                return None, str(e)

# ----------------------
# 🚀 Main Execution Logic
# ----------------------
async def query_agent(nl_input: str, verbose=False, session=None):
    """
    执行自然语言查询
    :param nl_input: 自然语言输入
    :param verbose: 是否显示详细信息
    :param session: 可选的外部数据库会话
    """
    ret_dic = {
        'err': 0,
        'sql': '',
        'sql_res': ''
    }
    
    if verbose: print("🎯 User input:", nl_input)

    # generate sql, attempt 1
    sql = await natural_language_to_sql(nl_input)
    ret_dic['sql'] = sql
    if verbose: print("\n🧠 SQL generated by GPT:", sql)

    # execute sql, attempt 1
    result, error = await execute_sql(sql, session)
    ret_dic["sql_res"] = result

    if error:
        # generate sql, attempt 2
        print("\n⚠️ Execution error, attempting to fix...")
        fixed_sql = await fix_sql_with_error(nl_input, sql, error)
        ret_dic['sql'] = fixed_sql
        if verbose: print("\n🔁 Fixed SQL:", fixed_sql)
        
        # execute sql, attempt 2
        result, error = await execute_sql(fixed_sql, session)
        ret_dic["sql_res"] = result

        if error:
            # define as failed if fix failed as well.
            ret_dic['err'] = 1
            if verbose: print("\n❌ Fix failed:", error)
        else:
            ret_dic['sql'] = fixed_sql
            if verbose: print("\n✅ Fix succeeded, results are as follows:\n", result)
    else:
        if verbose: print("\n✅ Execution succeeded, results are as follows:\n", result)
    
    if ret_dic['err'] != 0:
        ret_dic['sql'] = ''
        ret_dic['sql_res'] = ''
    
    return ret_dic

# ----------------------
# 🏁 CLI Entry Point
# ----------------------
async def run_agent():
    
    try:
        while True:
            nl_input = input("📝 Please input your natural language query:\n> ")
            
            output_to_console = True
            result = await query_agent(nl_input, verbose=output_to_console)
            print(result)
                # verbose = True: test mode;
                # verbose = False: interface mode
            print("Would you like to continue? (y/n)")
            choice = input("> ")
            
            if choice.lower() == 'n':
                print("Goodbye!")
                break
        
    finally:
        # Step 3: Restore proxy environment variables
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
        # Step 4: Properly release database resources
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(run_agent())
