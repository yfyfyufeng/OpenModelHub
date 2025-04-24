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

- model(model_id, model_name, param_num, media_type, arch_name, trainname)
  - model_id: Unique identifier
  - model_name: Name of the model
  - param_num: Number of parameters
  - media_type: Type of media the model handles, in { 'text', 'image', 'audio', 'video' }
  - arch_name: Architecture of the model, in { 'CNN', 'RNN', 'Transformer' }
  - trainname: Training method, in { 'pretrain', 'fine-tune', 'reinforcement learning' }

- model_tasks(model_id, task_name)
  - model_id: Model ID
  - task_name: Task the model performs, in { 'classification', 'detection', 'generation', 'regression' }

- cnn(model_id, module_num)
  - model_id: Foreign key to model
  - module_num: Number of convolutional modules (integer ≥ 0)

- module(model_id, conv_size, pool_type)
  - model_id: Foreign key to cnn
  - conv_size: Size of convolutional kernel (integer ≥ 0)
  - pool_type: Type of pooling, in { 'max', 'avg', 'min', 'other' }

- transformer(model_id, decoder_num, attn_size, up_size, down_size, embed_size)
  - model_id: Foreign key to model
  - decoder_num: Number of decoders (integer ≥ 0)
  - attn_size: Attention size (integer ≥ 0)
  - up_size: Upsample size (integer ≥ 0)
  - down_size: Downsample size (integer ≥ 0)
  - embed_size: Embedding size (integer ≥ 0)

- rnn(model_id, criteria, batch_size, input_size)
  - model_id: Foreign key to model
  - criteria: Loss function used (string)
  - batch_size: Batch size (integer ≥ 0)
  - input_size: Input size (integer ≥ 0)

- Dataset(ds_id, ds_name, ds_size, media, created_at)
  - ds_id: Dataset ID
  - ds_name: Dataset name
  - ds_size: Number of data samples (integer ≥ 0)
  - media: Type of media in the dataset, in { 'text', 'image', 'audio', 'video' }
  - created_at: Timestamp of creation
  - in the attributes, "ds" is the abbreviation of "dataset".

- Dataset_TASK(ds_id, task)
  - ds_id: Foreign key to dataset
  - task: Task associated with the dataset, in { 'classification', 'detection', 'generation', 'regression' }

- ds_col(ds_id, col_name, col_datatype)
  - ds_id: Foreign key to dataset
  - col_name: Name of a column in the dataset
  - col_datatype: Datatype of the column (e.g., 'int', 'string')

- user(user_id, user_name, password_hash, affiliate, is_admin)
  - user_id: User ID
  - user_name: Unique user name
  - password_hash: Encrypted password string
  - affiliate: Text name of organization
  - is_admin: Boolean flag { true, false }

- affil(affil_id, affil_name)
  - affil_id: Affiliation ID
  - affil_name: Name of the organization

- user_affil(user_id, affil_id)
  - Mapping of user to affiliation

- model_author(model_id, user_id)
  - Mapping of model to its author

- model_dataset(model_id, dataset_id)
  - Mapping of model to dataset

- user_ds(user_id, ds_id)
  - Mapping of user to dataset

# Synonym Handling:
- If the user query refers to a "language model", map it to "models where media_type = 'text'".

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

async def query_agent(nl_input: str, instance_type = 0, verbose = False, session = None):
    

    ret_dic = {
        'err': 0,
        'sql': '',
        'sql_res': ''
    }
    
    instance_dict = {
      1: 'model',
      2: 'Dataset',
      3: 'user',
      4: 'affiliation'
    }
    
    if verbose: print("🎯 User input:", nl_input)
    
    # preprocess user input: input from different pages
    if instance_type != 0 and instance_dict in instance_dict:
      nl_input += f"\nUser wants to search for {instance_dict[instance_type]} instances."
    elif instance_type == 0:
      nl_input += "\nUser wants to search across all tables including models, datasets, users and affiliations. Return results from all relevant tables."
    
    
    if verbose: print("🎯 Preprocessed input:", nl_input)

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
        result, error = await execute_sql(fixed_sql, sessoin)
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
            instance_type_choice = input("Please input the type of instance you'd like to search:\n0. no constraint\n1. model\n2. dataset\n3. user\n4. affiliation\n> ")
            if instance_type_choice not in {'0', '1', '2', '3', '4'}:
              instance_type_choice = '0'
              print("Invalid choice, automatically set to 0.")    
            instance_type = int(instance_type_choice)
            output_to_console = True
            result = await query_agent(nl_input, verbose=output_to_console, instance_type=instance_type)
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
