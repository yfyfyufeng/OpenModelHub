# CSC3170 Project: Database for Models and Datasets (Draft)

- Notation: different parts to complete.
  
  - [s] secure
  - [d] database 
  - [?] other todo list

## 1. Introduction and motivation

### Introduction

- Our project is a database for machine learning models and datasets. 
  
  - **Basic database operations:** It allows users to browse the information about the models and datasets, upload and download models and datasets. 

  - **Schema:** A variety of schemas is implemented. Apart from the basic schemas such as dataset, model, user, we also included schemas that are especially helpful for machine learning developers, such as the layer structure of different architecture of models (CNN, RNN, Transformer).
  
  - **GUI:** A beautifully designed graphic user interface is implemented, where users and administrators can perform multiple types of operations. 
  
  - **LLM:** An LLM agent is implemented, to translate user's natural language query into SQL language. User can also customize their query by selecting different tables and different fields. 
  
  - **Security:** Methods are implemented to protect data security, 

### Motivation

- We are motivated by [huggingface](https://huggingface.co/), one of the most influential platform in the AI community that facilitates the sharing and collaboration of machine learning models and datasets. As the number of machine learning models and datasets continues to grow rapidly, there is a pressing need for a structured and efficient way to manage these resources.

- When using a machine learning database with a considerable amount of data, it may be challenging for users to find a model or dataset that suits their requirement by traditional seache methods, or by directly inputting SQL, which may be technically challenging for general users. 

- Therefore, we implement an agent to assists the user's queries, enhancing flexibility for users to perform customized operations, as well as efficiency especially when the user's need is too complicated to be manually written into SQL.

- Furthermore, the increasing concern over data security and privacy necessitates more robust solutions to protect valuable information. [s]

##### How to run our code?

- **Step 1-3 has to be done ONLY when running it at the first time; if it's not the first time, you can skip 1-3, and also can skip 4 if you don't need to initialize the database.**

1. Install dependencies according to `requirement.txt` [?]
2. Create an `.env` file at the root directory of the project, and add the following lines to it (repalce `$your_api_key` and `$your_base_url` with your own values):
   ```bash
   # -----database-----
    DB_USERNAME=root
    DB_PASSWORD=123
    DB_HOST=0.0.0.0
    DB_PORT=3306
    TARGET_DB=openmodelhub
   # -----agent----
   API_KEY=$your_api_key
   BASE_URL=$your_base_url
   ```
3. Test connection by running `database/db_connection_check.py`.
4. Initialize the database with the records stored in `database/records/demo.json`, by running:
  
  ```shell
  database/load_data.py
  ```
  
  - then you'll be asked to choose a .json file stored in `database/records` to intialize it; just choose `demo.json`.

5. Run the GUI:

  ```shell
  streamlit run frontend/app.py
  ```

## 2. Design and implementation

### Project Structure

- our project is composed of the following components:
  1. Database. 
  2. Data.
  3. Frontend.
  4. Agent.
  5. Security.
  6. Data analysis.

#### 1. Database

- [d][THE WHOLE PART needs fact-checking!! whether my description is accurate?] 

##### Schema Design

- Our database follows the relational model and the 4th normal form.
- Our schema are as follows:

- [d] [please insert a markdown format table here to show the schema. can be generated from our slides.]

- [?] llm optimized design

##### Implmentation

- In `database/database_schema.py`, schemas are represented by python classes. 
- In `database/database_interface.py`, we have encapsulated interfaces to perform SQL operations safely. Therefore, in other programs where we have to execute SQL, we can call an encapsulated functions instead of executing the SQL operations directly.

#### 2. Data.

##### Initialization

- We created a set of records to initialize our database; although more records can be inserted to or deleted from the database during use.
- The records used to initialize 


### Detailed implementation

#### 1. 
