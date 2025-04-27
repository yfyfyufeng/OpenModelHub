# CSC3170 Project: Database for Models and Datasets (Draft)

- Notation: different parts to complete.
  
  - [s] secure

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

-  Furthermore, the increasing concern over data security and privacy necessitates more robust solutions to protect valuable information. [s]


## 2. Design and implementation

### Project Structure

- our project is composed of the following components:

    1. **Database.** Including database schema, and encapsulated interfaces for porforming SQL operations.
    
    2. **Data.** We have created 
    
    3. **Frontend.**.

### Detailed implementation

#### 1. 
