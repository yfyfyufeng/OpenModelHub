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
- model_tasks(model_id, task_name)
- cnn(model_id, module_num)
- module(model_id, conv_size, pool_type)
- transformer(model_id, decoder_num, attn_size, up_size, down_size, embed_size)
- rnn(model_id, criteria, batch_size, input_size)
- dataset(ds_id, ds_name, ds_size, media, task, created_at)
- ds_col(ds_id, col_name, col_datatype)
- user(user_id, user_name, password_hash, affiliate, is_admin)
- affil(affil_id, affil_name)
- user_affil(user_id, affil_id)
- model_author(model_id, user_id)
- model_dataset(model_id, dataset_id)
- user_ds(user_id, ds_id)

# Synonym Handling:
If the user query refers to a "language model", but no such table or attribute exists,
you should map "language model" to "models where media_type = 'text'".


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
async def execute_sql(sql: str):
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
async def query_agent(nl_input: str, verbose = False):
    
    ret_dic = {
        'err' : 0,
        'sql' : '',
        'sql_res' : ''
    }
    
    if verbose: print("🎯 User input:", nl_input)

    # generate sql, attempt 1
    sql = await natural_language_to_sql(nl_input)
    ret_dic['sql'] = sql
    if verbose: print("\n🧠 SQL generated by GPT:", sql)

    # execute sql, attempt 1
    result, error = await execute_sql(sql)
    ret_dic["sql_res"] = result

    if error:
        
        # generate sql, attempt 2
        print("\n⚠️ Execution error, attempting to fix...")
        fixed_sql = await fix_sql_with_error(nl_input, sql, error)
        ret_dic['sql'] = fixed_sql
        if verbose: print("\n🔁 Fixed SQL:", fixed_sql)
        
        # execute sql, attempt 2
        result, error = await execute_sql(fixed_sql)
        ret_dic["sql_res"] = result

        if error:
            # define as failed if fix failed as well.
            ret_dic['err'] = 1
            if verbose: print("\n❌ Fix failed:", error)
        
        else:
            ret_dic['sql'] = fixed_sql
            if verbose: print("\n✅ Fix succeeded, results are as follows:\n",result)
    else:
        if verbose: print("\n✅ Execution succeeded, results are as follows:\n",result)
    
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
