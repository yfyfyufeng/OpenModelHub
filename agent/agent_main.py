import os
import openai
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# --------------------
# 🔧 环境配置
# --------------------
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL")

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
TARGET_DB = os.getenv("TARGET_DB")

DATABASE_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}"

# 初始化数据库连接
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

# ----------------------
# 📘 Schema-aware Prompt
# ----------------------
SYSTEM_PROMPT = """你是一个 SQL 生成器，请根据自然语言请求生成 MySQL 查询语句，查询的数据结构如下：

- model(model_id, model_name, param_num, media_type, arch_name)
- model_tasks(model_id, task_name)
- cnn(model_id, module_num)
- transformer(model_id, decoder_num, attn_size, up_size, down_size, embed_size)
- rnn(model_id, criteria, batch_size, input_size)
- dataset(ds_id, ds_name, ds_size, media, task)
- ds_col(ds_id, col_name, col_datatype)
- user(user_id, user_name, affiliate)
- affil(affil_id, affil_name)
- user_affil(user_id, affil_id)
- model_author(model_id, user_id)
- model_dataset(model_id, dataset_id)
- user_ds(user_id, ds_id)

只返回 SQL 查询语句，不要添加其他解释性文字。"""

# ----------------------
# 🔁 生成 SQL
# ----------------------
def natural_language_to_sql(nl_input: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_input}
        ],
        temperature=0,
    )
    return response.choices[0].message["content"].strip("` \n")

# ----------------------
# ❌ 错误自动修复
# ----------------------
def fix_sql_with_error(nl_input: str, original_sql: str, error_msg: str) -> str:
    fix_prompt = f"""
原始自然语言请求是：
{nl_input}

生成的 SQL 是：
{original_sql}

执行时出现了错误：
{error_msg}

请修复这个 SQL 查询，返回正确语法的 SQL 查询语句。
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fix_prompt}
        ],
        temperature=0,
    )
    return response.choices[0].message["content"].strip("` \n")

# ----------------------
# 🧪 执行 SQL
# ----------------------
def execute_sql(sql: str):
    session = SessionLocal()
    try:
        result = session.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()
        return [dict(zip(columns, row)) for row in rows], None
    except Exception as e:
        return None, str(e)
    finally:
        session.close()

# ----------------------
# 🚀 主逻辑
# ----------------------
def query_agent(nl_input: str):
    print("🎯 用户输入：", nl_input)

    # 第一次生成 SQL
    sql = natural_language_to_sql(nl_input)
    print("\n🧠 GPT生成的SQL：", sql)

    result, error = execute_sql(sql)

    # 如果出错，尝试修复
    if error:
        print("\n⚠️ 执行出错，尝试自动修复中...")
        fixed_sql = fix_sql_with_error(nl_input, sql, error)
        print("\n🔁 修复后的SQL：", fixed_sql)
        result, error = execute_sql(fixed_sql)

        if error:
            print("\n❌ 修复仍失败：", error)
        else:
            print("\n✅ 修复成功，结果如下：")
            print(result)
    else:
        print("\n✅ 执行成功，结果如下：")
        print(result)

# ----------------------
# 🏃 入口
# ----------------------
if __name__ == "__main__":
    nl_input = input("📝 请输入你的自然语言查询：\n> ")
    query_agent(nl_input)
