# MySQL连接常见问题

## 测试流程

1. 运行`db_operations_test.py`
2. 成功之后运行`db_connection_check.py`

## 可能的异常情况（主要是connection_check部分）

### 1. 报错 `Access denied for user 'root'@'localhost'`

- 报错：
    
    ```
    🔍 正在使用以下配置尝试连接数据库：
    Host: 0.0.0.0
    Port: 3306
    User: root
    Database: openmodelhub
    ❌ [pymysql] 无法连接： (1698, "Access denied for user 'root'@'localhost'")
    ❌ [aiomysql] 无法连接： (1698, "Access denied for user 'root'@'localhost'")
    ```

- 原因：没有给root用户创建密码。
- 解决方案：给root用户设置密码。登陆sql之后输入`ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123';`。

- 密码必须设置为`123`,为了和这个项目的环境一致。否则[运行时出现报错](). 如果一开始的密码不是123, 可以用上述指令重新设置密码。
- 设置密码时可能出现的问题：[`Your password does not satisfy the current policy requirements`](#2-报错-your-password-does-not-satisfy-the-current-policy-requirements)

```
mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'yourpassword';
ERROR 1819 (HY000): Your password does not satisfy the current policy requirements
```

### 2. 报错 `Your password does not satisfy the current policy requirements`

```
mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123';
ERROR 1819 (HY000): Your password does not satisfy the current policy requirements
```

- 原因：MySQL中的`validate_password` component只允许强密码。
- 解决：
  
  - 最好的方式是一开始安装的时候就不要enable这个validate_password. 以ubuntu系统为例，在执行这个操作的时候，第一步就是validate_password, 要选n
    
    ```
    sudo mysql_secure_installation
    ```
  
  - 如果已经enable了，可以进入mysql并取消。以ubuntu系统为例：

    ```sql
    UNINSTALL COMPONENT 'file://component_validate_password';
    ```

### 3. 报错： `"Access denied for user 'root'@'localhost' (using password: YES)")`

```
🔍 正在使用以下配置尝试连接数据库：
Host: 0.0.0.0
Port: 3306
User: root
Database: openmodelhub
❌ [pymysql] 无法连接： (1045, "Access denied for user 'root'@'localhost' (using password: YES)")
❌ [aiomysql] 无法连接： (1045, "Access denied for user 'root'@'localhost' (using password: YES)")
```

- 原因：密码错误（不过已经在使用密码了）。请根据[设置root用户密码](#1-报错-access-denied-for-user-rootlocalhost)部分的指示设置密码为123.
