所有 **schema 表（class）与字段（attribute）** 的 **domain 是否被定义**，并给出说明：

---

## ✅ 1. `Model` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| model_id      | Integer (PK) | ✅ 自增主键       | 主键，自增长 |
| model_name    | String(50)   | ❌               | 任意字符串 |
| param_num     | BIGINT       | ❌               | 未限制范围 |
| media_type    | String(50)   | ❌               | 未限定，例如 text, image |
| arch_name     | `Enum(ArchType)` | ✅         | 限定为 CNN, RNN, TRANSFORMER |

---

## ✅ 2. `ModelTask` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| model_id      | FK           | ✅ 外键约束       | 引用 `model.model_id` |
| task_name     | String(50)   | ❌               | 未设定 task 的取值范围 |

---

## ✅ 3. `CNN`, `Module` 表

### CNN:

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| model_id      | FK           | ✅               | 主键 + 外键 |
| module_num    | Integer      | ❌               | 未限制 |

### Module:

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| model_id      | FK           | ✅               | 外键 |
| conv_size     | Integer      | ❌               | 任意整数 |
| pool_type     | String(20)   | ❌（规范使用）   | 示例中使用了 "max", "avg"，未强制限定 |

---

## ✅ 4. `RNN` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| model_id      | FK/PK        | ✅               | 引用 model |
| criteria      | String(50)   | ❌               | 比如 "MSE"，但未限定 |
| batch_size    | Integer      | ❌               | 未限制 |
| input_size    | Integer      | ❌               | 未限制 |

---

## ✅ 5. `Transformer` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| model_id      | FK/PK        | ✅               | 引用 model |
| decoder_num   | Integer      | ❌               | 任意整数 |
| attn_size     | Integer      | ❌               | 同上 |
| up_size       | Integer      | ❌               | 同上 |
| down_size     | Integer      | ❌               | 同上 |
| embed_size    | Integer      | ❌               | 同上 |

---

## ✅ 6. `Dataset` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| ds_id         | PK           | ✅               | 主键 |
| ds_name       | String(255)  | ❌               | 任意字符串 |
| ds_size       | Integer      | ❌               | 任意整数 |
| media         | String(50)   | ❌               | 示例中为 image，但未限定 |
| task          | Integer      | ❌               | 没有说明与 task 表关联 |
| created_at    | DateTime     | ✅ 默认当前时间   | 自动生成时间戳 |

---

## ✅ 7. `DsCol` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| ds_id         | FK           | ✅               | 数据集外键 |
| col_name      | String(50)   | ❌               | 任意字符串 |
| col_datatype  | String(20)   | ❌（测试中使用了string/int） | 未限制 |

---

## ✅ 8. `User` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| user_id       | PK           | ✅               | 主键 |
| user_name     | String(50)   | ❌               | 唯一，但无值域限定 |
| password_hash | String(100)  | ❌               | 任意字符串 |
| affiliate     | String(50)   | ❌               | 任意字符串 |
| is_admin      | Boolean      | ✅ 仅 True/False | 逻辑布尔 |

---

## ✅ 9. `Affil` 表

| Attribute     | Type         | Domain Defined? | 说明 |
|---------------|--------------|------------------|------|
| affil_id      | PK           | ✅               | 自增主键 |
| affil_name    | String(100)  | ❌               | 任意字符串 |

---

## ✅ 10. 中间表（多对多）

这些中间表主要用于建立关联，属性均为外键：

- `UserAffil`
- `UserDataset`
- `ModelAuthor`
- `ModelDataset`

| 表名            | Attribute        | Domain Defined? |
|------------------|------------------|------------------|
| UserAffil        | user_id, affil_id| ✅ 外键限制       |
| UserDataset      | user_id, ds_id   | ✅ 外键限制       |
| ModelAuthor      | model_id, user_id| ✅ 外键限制       |
| ModelDataset     | model_id, dataset_id | ✅ 外键限制   |

---

## 📌 总结表格

| 表名          | 属性名           | Domain是否定义 | 备注 |
|---------------|------------------|----------------|------|
| Model         | arch_name        | ✅ Enum        | CNN/RNN/Transformer |
| ModelTask     | task_name        | ❌             | 无枚举 |
| CNN/Module    | pool_type        | ❌             | 示例中约定使用 max/avg |
| RNN           | criteria         | ❌             | 无限定 |
| Dataset       | media            | ❌             | 示例 image/text |
| DsCol         | col_datatype     | ❌             | 示例 string/int |
| User          | is_admin         | ✅ Boolean     | True / False |
| 其他主键/外键 | 所有ID           | ✅             | 外键或自增主键 |

