# Demonstration

## Run the program

### 1. backend: import data

- Show how we can initialize the database by importing the json file

- `demo.json` is the previous `db_with_pswd.json`

```shell
python database/load_data.py
```

### 2. frontend: run program

```shell
streamlit run frontend/app.py
```

1. user types
   1. common user login
   2. common user register and login
      ![user register](material/regis.png)
   3. admin login: has some pages that common users don't have

   | type    | user                                    | admin                                     |
   |---------|-----------------------------------------|-------------------------------------------|
   | sidebar | ![user sidebar](material/side_user.png) | ![admin sidebar](material/side_admin.png) | 

2. page types
    1. Home
       - export and download data
       - ![homepage download data](material/home_dl.png) 
    2. Model Repository
       - search with/without specifying the instance
       - ![classified model query](material/llm_cls.png)
       - [LLM assisted search](#llm-assisted-search)
       - upload model
       - ![upload](material/model_upload.png)
       - view details; paging; **view details: show 2 tables**
       - ![view details](material/model_detail.png)

   3. Datasets
   4. **(Admin Privilege)** User Management

      - create user ![create user](material/mng_create.png)
      - [edit user](material/page_userMng.pdf)

   5. **(Admin Privilege)** Data Insights

      - (presenter: Linyong Gan)
      - ![page1](material/anlys_p1.png)
      - ![page2](material/anlys_p2.png)

#### LLM assisted search

1. search according to architecture: transfomer models

    - result in the table: ...
        ![query: LLM](material/llm_tf.png)
    - sql query is also available:
        ![query details in SQL](material/llm_sql.png)

2. compare classified and not classified

    |instance type|model|dataset|user|
    |----------|------|-------|-----|
    |query: show all names|![model names](material/llm_cls.png)|![dataset names](material/llm_cls_ds.png)|![user names](material/llm_cls_user.png)

3. more complicated search: ranking

    - query: top 10 users with the most published datasets    
    ![rank users](material/llm_rank.png)

# todo list

## urgent

- [x] translate all code
- [ ] "LLM refined schema"
- [ ] powerpoint
- [ ] code demo

## not urgent

- [ ] report + each persons' part
- [ ] report zip, pdf

