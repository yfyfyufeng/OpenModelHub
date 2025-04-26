from pathlib import Path
import asyncio
import os
import sys
import pandas as pd
from openai import AsyncOpenAI

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
from frontend.database_api import db_export_all_data

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    # proxies= None
)


def deal_model(df_model):
    output = {}
    attributes = ["media_type", "arch_name", "trainname"]
    types = {"media_type":["audio", "image", "text", "video"],
             "arch_name": ["CNN", "RNN", "Transformer"], 
             "trainname": ["Trainname.FINETUNE", "Trainname.PRETRAIN", "Trainname.RL"],
            }
    
    for attr in attributes:
        output[attr] = {}
        for ty in types[attr]:
            output[attr][ty] = len(df_model[df_model[attr] == ty])

    output["media_task_relation"] = {}
    tasks = ["classification", "detection", "generation", "regression", "segmentation"]
    for ty in types["media_type"]:
        output["media_task_relation"][ty] = {}
        for task in tasks:
            output["media_task_relation"][ty][task] = 0
    for i in range(len(df_model)):
        for task in df_model.at[i, "task"]:
            output["media_task_relation"][df_model.at[i, "media_type"]][task] += 1
    
    model_param_num = df_model["param_num"]
    output["param_num"] = {"max": model_param_num.max(), "min": model_param_num.min(), "mean": model_param_num.mean(), "std": model_param_num.std()}

    comment = ai_summary("please summary these data respectively", output)
    output["comment"] = comment
    return output

def deal_dataset(df_dataset):
    output = {}
    types = {"media_type":["audio", "image", "text", "video"],
            "arch_name": ["CNN", "RNN", "Transformer"], 
            "trainname": ["Trainname.FINETUNE", "Trainname.PRETRAIN", "Trainname.RL"],
            }
    
    output["media_task_relation"] = {}
    tasks = ["classification", "detection", "generation", "regression", "segmentation"]
    for ty in types["media_type"]:
        output["media_task_relation"][ty] = {}
        for task in tasks:
            output["media_task_relation"][ty][task] = 0
    for i in range(len(df_dataset)):
        for task in df_dataset.at[i, "task"]:
            output["media_task_relation"][df_dataset.at[i, "media"]][task] += 1

    ds_size = df_dataset["ds_size"]
    output["ds_size"] = {"max": ds_size.max(), "min": ds_size.min(), "mean": ds_size.mean(), "std": ds_size.std()}
    
    comment = ai_summary("please summary these data respectively", output)
    output["comment"] = comment

    return output

def deal_user(df_user):
    count = df_user.groupby("affiliate", as_index = False).count()[["affiliate", "user_id"]]
    count = count.rename(columns = {"user_id": "count"})
    print(count)
    return count

async def ai_summary(SYSTEM_PROMPT, nl_input):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_input}
        ],
        temperature=0,
    )
    return response.choices[0].message.content

def data_ins():
    json = db_export_all_data()
    output = {"model":{}, "dataset":{}, "user":{}}
    affil = json["affiliation"]
    user = json["user"]
    dataset = json["dataset"]
    model = json["model"]

    df_model = pd.DataFrame(model)
    df_dataset = pd.DataFrame(dataset)
    df_user = pd.DataFrame(user)
    df_user = df_user[df_user["is_admin"] == False]
    print(df_user)

    output["model"] = deal_model(df_model)
    output["dataset"] = deal_dataset(df_dataset)
    output["user"] = deal_user(df_user)
    
    #print(len(df_model_audio), len(df_model_image), len(df_model_text), len(df_model_video))
    return output


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(data_ins())
    finally:
        loop.close()