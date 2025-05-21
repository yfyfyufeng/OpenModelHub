# OpenModelHub

OpenModelHub is an open-source machine learning model and dataset management platform, inspired by [HuggingFace](https://huggingface.co/). This project provides a complete solution for storing, managing, and sharing machine learning models and datasets.

## 🌟 Key Features

- **Database Management**: Supports basic CRUD operations for models and datasets
- **Intelligent Search**: Integrated LLM agent supporting natural language queries
- **Security Mechanisms**: Implements hybrid encryption, access control, and audit trails
- **User Interface**: Beautiful Streamlit interface with different permissions for users and administrators
- **Data Analytics**: Provides data insights and visualizations
- **File Management**: Supports model and dataset upload/download

## 🚀 Quick Start

### Prerequisites

- Python 3.x
- Go 1.x (for security module)
- MySQL 8.x
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/OpenModelHub.git
cd OpenModelHub
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:
   Create a `.env` file in the project root directory with the following content:

```bash
# -----database-----
DB_USERNAME=$your_DB_USERNAME
DB_PASSWORD=$your_DB_PASSWORD
DB_HOST=0.0.0.0
DB_PORT=3306
TARGET_DB=openmodelhub

# -----agent----
API_KEY=$your_api_key
BASE_URL=$your_base_url
```

### Running the Project

1. Using the startup script (recommended):

```bash
python start.py
```

This will automatically:

- Initialize the database
- Start the security service
- Launch the web interface

2. Manual startup:

```bash
# 1. Initialize database
python database/load_data.py
# Choose demo.json for initialization

# 2. Start security service
cd security
go run main.go

# 3. Launch web interface
streamlit run frontend/app.py
```

## 👥 User Guide

### Login System

- Admin account: username: `admin`, password: `admin`
- Regular users: Can register new accounts, default password is `admin`

### Main Features

1. **Model Management**

   - Browse model repository
   - Upload new models
   - Download models
   - LLM-assisted search

2. **Dataset Management**

   - Browse datasets
   - Upload new datasets
   - Download datasets
   - Dataset analysis

3. **User Management** (Admin only)

   - View user list
   - Create/edit users
   - Permission management

4. **Data Insights** (Admin only)
   - Model analysis
   - Dataset analysis
   - User analysis

## 🔒 Security Features

- Hybrid encryption (AES-CBC + RSA)
- Argon2Key password hashing
- Role-based access control
- Secure data sharing mechanism
- Comprehensive audit logging

## 🏗️ Project Structure

```
OpenModelHub/
├── database/           # Database related code
│   ├── records/       # Sample data
│   └── schema/        # Database schema definitions
├── frontend/          # Streamlit frontend
├── security/          # Go security module
├── doc/              # Documentation
└── requirements.txt   # Python dependencies
```

## 📊 Database Design

The project uses a relational database with the following main entities:

- Model
- Dataset
- User
- Affiliation
- Task

For detailed database schema, please refer to the [Database Design Documentation](doc/demo_and_report/report-UNFINISHED.md#21-database)

## 🤝 Contributing

Issues and Pull Requests are welcome! Before submitting a PR, please ensure:

1. Code follows project standards
2. Necessary tests are added
3. Documentation is updated

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) - Project inspiration
- All project contributors

## 📚 References

- Argon2: Next Generation Password Hashing Algorithm
- HMAC: Keyed-Hashing for Message Authentication
- RSA: Public-Key Cryptosystem
- More references can be found in the [Project Report](doc/demo_and_report/report-UNFINISHED.md#4-references)
