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

### Model

| schema        | model_id (PK) | model_name | param_num | media_type (FK) | arch_id (FK) | train_ID (FK) | task_id (FK)     |
|---------------|---------------|------------|-----------|------------------|---------------|----------------|------------------|
| data type     | int           | varchar    | int       | multivalued,int  | multivalued,int | multivluaed,int | multivalued, int |
| example       | 000000001     | llama_2_7b | 6.74B     | 01               | 01            | 1              | {01,03}          |

#### -- media

| media_type | media_name  |
|------------|-------------|
| 01         | language    |
| 02         | vision      |
| 03         | audio       |
| 04         | multimodal  |

#### -- arch

| arch_id | arch_name   |
|---------|-------------|
| 01      | transformer |
| 02      | CNN         |
| 03      | RNN         |
| 04      | …           |

#### -- task

| task_id  |
|----------|
| 01       | classification |
| 02       | regression     |
| 03       | generation     |
| 04       | …              |

#### -- train_ID

| train_ID | train_name |
|----------|------------|
| 01       | pretrained |
| 02       | finetuned  |
| …        | …          |

#### ---- model_lang (could also be model_vis ...)

| model_id (PK) | content_len |
|---------------|-------------|
| 00000001      | 4k          |

#### ---- model_tf (tf short for transformer)

*(乱写的)*  
| model_id | decoder_layer | head_number | head_size |
|----------|----------------|-------------|-----------|
|          | 32             | 24          | 2048      |

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