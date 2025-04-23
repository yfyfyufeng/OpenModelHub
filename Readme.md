# OpenModelHub - A Secure AI Resource Platform

## 1. Introduction & Motivation  

### 1.1 Organizational Context  
We propose building OpenModelHub, an AI resource-sharing platform inspired by Hugging Face, designed for:  
- Researchers to publish papers and ML models  
- Developers to share/download datasets  
- Enterprises to securely access commercial models  
- General Users to interactively search technical resources  

### 1.2 Value Proposition  

| Module          | Key Features                                          | Course Alignment                |
|-----------------|--------------------------------------------------------|---------------------------------|
| Database System | Manage Model/Dataset/Paper/User entities              | ER Diagram, Relational Schema, SQL Queries |
| Security Layer  | Encryption&Decryption, Access Control based on Key Derivation | Security & Privacy Bonus       |
| LLM Agent       | Natural Language to SQL Converter                     | LLM+DB Bonus                    |
| Web Interface   | Responsive Dashboard                                  | Web Design                      |

---

## 2. Design and Implementation  

### 2.1 Database Design  

#### 2.1.1 Entity-Relationship Diagram  

```
erDiagram  
    USER ||--o{ MODEL : "uploads"  
    USER ||--o{ DATASET : "owns"  
    PAPER }o--|| MODEL : "references"  
    MODEL ||--o{ DATASET : "uses"  
    
    USER {  
        UUID user_id PK  
        BYTEA encrypted_email  
        ENUM(admin,developer,guest) role  
        TIMESTAMP created_at  
    }  
    
    MODEL {  
        UUID model_id PK  
        VARCHAR(255) name  
        BYTEA encrypted_weights  
        ENUM(mit,apache,proprietary) license  
        INT download_count  
    }  
```

#### 2.1.2 Relational Schema  

**-- Implement 3NF Normalization**  
```sql
CREATE TABLE users (  
    user_id UUID PRIMARY KEY,  
    email BYTEA,  
    role VARCHAR(10) NOT NULL CHECK (role IN ('admin', 'developer', 'guest')),  
    public_key TEXT -- For field-level encryption  
);
```

**-- Handle Multi-valued Dependency (Model-Tags)**  
```sql
CREATE TABLE model_tags (  
    model_id UUID REFERENCES models(model_id),  
    tag VARCHAR(50) NOT NULL,  
    PRIMARY KEY (model_id, tag)  
);
```

Example: [https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)

---

#### **Model**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **model_id (PK)**   | A unique identifier for the model (Primary Key).                                                                       | int              | 1001         |
| **model_name**      | Name of the model.                                                                                                     | varchar          | gpt_4_13b    |
| **param_num**       | The number of parameters in the model.                                                                                  | int              | 175000000000 |
| **media_type**      | Foreign Key that links to a table that defines the type of media the model deals with.                                  | varchar          | text         |
| **arch_name (FK)**  | Foreign Key linking to the architecture name.                                                                          | varchar          | transformer  |
| **train_name (FK)** | Foreign Key linking to the training process or dataset used for training.                                              | varchar          | pretrained   |
 | **param**           |   binary parameter of the model|                                                                                       |largebinary | 0x...         |  

#### **ModelTask**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **model_id (FK)**    | Foreign Key referencing **model_id** in the model table.                                                                | int              | 1001         |
| **task_name**        | The task associated with the model (e.g., classification, translation).                                                 | varchar          | classification |

#### **Transformer**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **model_id (PK, FK)**| Primary Key, Foreign Key referencing **model_id** in the model table.                                                   | int              | 1001         |
| **decoder_num**      | Number of decoders used in the model.                                                                                   | int              | 12           |
| **attn_size**        | Size of the attention mechanism used in the model.                                                                    | int              | 64           |
| **up_size**          | The upsampling size in the model.                                                                                      | int              | 256          |
| **down_size**        | The downsampling size in the model.                                                                                    | int              | 64           |
| **embed_size**       | The size of the embedding layer in the model.                                                                          | int              | 128          |

#### **CNN**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **model_id (PK, FK)**| Primary Key, Foreign Key referencing **model_id** in the model table.                                                   | int              | 1001         |
| **module_num**       | The number of modules (e.g., convolution layers) in the model.                                                         | int              | 3            |

#### **Module**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **model_id (FK)**    | Foreign Key referencing **model_id** in the cnn table.                                                                | int              | 1001         |
| **conv_size**        | The size of the convolution layers in the model.                                                                      | int              | 3            |
| **pool_type**        | The type of pooling used in the model (e.g., max pooling, average pooling).                                            | varchar          | max          |

#### **RNN**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **model_id (PK, FK)**| Primary Key, Foreign Key referencing **model_id** in the model table.                                                   | int              | 1001         |
| **criteria**         | The training criteria or loss function used by the RNN model.                                                           | varchar          | cross_entropy|
| **batch_size**       | The number of data samples processed together in one pass through the model (used in training).                         | int              | 64           |
| **input_size**       | The size of the input data (e.g., number of features for each data point).                                              | int              | 256          |

#### **Dataset**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **ds_id (PK)**       | Unique identifier for the dataset.                                                                                     | int              | 1            |
| **ds_name**          | The name of the dataset (e.g., "Coco", "ImageNet").                                                                    | varchar          | Coco         |
| **ds_size**          | The size of the dataset (e.g., number of images or data points).                                                       | int              | 1000000      |
| **media**            | The type of media in the dataset (e.g., image, text, video).                                                           | varchar          | image        |
| **created_at**       | The date and time when the dataset was created.                                                                | timestamp        | 2025-04-01   |

#### **Dataset_TASK**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **ds_id (FK)**       | Foreign Key referencing **ds_id** in the dataset table.                                                               | int              | 1            |
| **task**             | The task associated with the dataset (e.g., classification, segmentation).                                             | int              | 1            |

#### **DsCol**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **ds_id (PK)**       | Foreign Key referencing **ds_id** in the dataset table.                                                               | int              | 1            |
| **col_name**         | Name of the column within the dataset (e.g., "image_id", "label").                                                      | varchar          | image_id     |
| **col_datatype**     | The data type of the column (e.g., integer, float, text).                                                              | varchar          | integer      |

#### **user**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **user_id (PK)**     | Unique identifier for the user.                                                                                         | int              | 123090342    |
| **user_name**        | The name of the user (e.g., "JohnDoe").                                                                                  | varchar          | JohnDoe      |
| **model_id (FK)**    | Foreign Key referencing **model_id** in the model table.                                                               | int              | 1001         |
| **ds_id (FK)**       | Foreign Key referencing **ds_id** in the dataset table.                                                                | int              | 1            |
| **affiliate**        | A reference to the affiliate company or group the user is associated with.                                             | varchar          | AffiliateX   |
| **password_hash**     || The hashed password of the user.                                                                                         | varchar          | $2b$12$...   |
| **is_admin**          | | Boolean flag indicating if the user has admin privileges.                                                             | boolean          | true         |   

#### **Affil**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **affil_id (PK)**    | Unique identifier for the affiliate.                                                                                   | int              | 1            |
| **affil_name**       | Name of the affiliate organization.                                                                                     | varchar          | AffiliateX   |

#### **UserAffil**
| **Attribute**       | **Description**                                                                                                       | **Data Type**    | **Example**  |
|---------------------|-----------------------------------------------------------------------------------------------------------------------|------------------|--------------|
| **user_id (FK)**     | Foreign Key referencing **user_id** in the user table.                                                                 | int              | 123090342    |
| **affil_id (FK)**    | Foreign Key referencing **affil_id** in the affil table.                                                               | int              | 1            |

#### **UserDataset**
| **Attribute**   | **Description**                                                                  | **Data Type**   | **Example** |
|-----------------|----------------------------------------------------------------------------------|-----------------|-------------|
| **user_id (FK)** | 外键，引用 **user** 表中的 `user_id`                                             | int             | 123090342   |
| **ds_id (FK)**   | 外键，引用 **dataset** 表中的 `ds_id`                                           | int             | 1           |

#### **ModelAuthor**
| **Attribute**   | **Description**                                                                  | **Data Type**   | **Example** |
|-----------------|----------------------------------------------------------------------------------|-----------------|-------------|
| **model_id (FK)**| 外键，引用 **model** 表中的 `model_id`                                           | int             | 1001        |
| **user_id (FK)** | 外键，引用 **user** 表中的 `user_id`                                             | int             | 123090342   |

#### **ModelDataset**

| **Attribute**   | **Description**                                                                  | **Data Type**   | **Example** |
|-----------------|----------------------------------------------------------------------------------|-----------------|-------------|
| **model_id (FK)**| 外键，引用 **model** 表中的 `model_id`                                           | int             | 1001        |
| **ds_id (FK)**   | 外键，引用 **dataset** 表中的 `ds_id`                                           | int             | 1           |

---

### 2.2 Security Implementation  
File access key only distributed to authorized users.

---

### 2.3 LLM Agent Implementation  

#### 2.3.1 NL-to-SQL Conversion Pipeline  

**User Input**:  
> "Show top 10 CV models with most downloads last month"  

**↓**  

**LLM Parsing → Intermediate Representation:**  
```json
{  
  "target": "models",  
  "filters": [  
    {"field": "category", "operator": "=", "value": "CV"},  
    {"field": "upload_date", "operator": ">", "value": "2025-04-01"}  
  ],  
  "sorting": {"field": "download_count", "order": "DESC"},  
  "pagination": {"limit": 10}  
}
```

**↓**  

**Generated SQL:**  
```sql
SELECT * FROM models  
WHERE category='CV' AND upload_date > '2025-04-01'  
ORDER BY download_count DESC LIMIT 10;
```

#### 2.3.2 Prompt Engineering Strategy  

- Schema-aware Few-shot Learning: Provide 5 NL-SQL pairs per entity  
- SQL Injection Prevention: Strict output validation using regex patterns  
- Error Correction: Implement GPT-4 based query debugging  

---

## 3. Conclusion and Self-evaluation  

### 3.1 Technology Stack  

| Component | Technologies                        |
|----------|--------------------------------------|
| Backend  | Python Flask, SQLAlchemy, PostgreSQL |
| Frontend | React.js, Ant Design, Chart.js       |
| LLM      | LangChain, OpenAI GPT-3.5 Turbo      |
| Security | Sync/Async Cryptography, Security Schema Design |

---

### 3.2 Milestones  

| Phase                  | Deadline | Deliverables               |
|------------------------|----------|----------------------------|
| Database implementation|          |                            |
| LLM integration        |          |                            |
| Frontend implementation|          |                            |

---

### 3.3 Self-evaluation  

---

### 3.4 Anticipated Challenges  

---

## 4. References  

---

## 5. Appendices  

### 5.1. Team Structure  

| Members               | Responsibilities    | Key Technologies |
|-----------------------|---------------------|------------------|
|                       | Database Implementation |                |
|                       | Security Module         |                |
|                       | LLM Agent               |                |
|                       | Frontend Development    |                |

---

## Current Work
| Task     | Member   | Status |
|----------|----------|--------|
| Model    | lyf, gly | R      |
| Dataset  | lyx      | R      |
| User     | zzr      | R      |
| Agent    | tym      | R      |
| Security | zsh      | PD     |

## Schedule
Meet in 4/18
