import pandas as pd
import mysql.connector
from mysql.connector import Error


def create_connection():
    """ 创建数据库连接 """
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="/GAMEmode1",
            database="test_data"
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection, query):
    """ 执行SQL查询 """
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def read_excel_file(filename):
    """ 读取Excel文件并返回DataFrame """
    df = pd.read_excel(filename, engine='openpyxl')
    return df


def insert_data(connection, data):
    """ 插入数据到MySQL数据库 """
    cursor = connection.cursor()

    # 替换 DataFrame 中的 NaN/None 值为 None，以便在 SQL 语句中被解释为 NULL
    data = data.where(pd.notnull(data), None)

    # 使用正确的占位符
    sql = ("INSERT INTO strategies (指标, 异常区间, 策略组, 部门A, 约束准则A, 评分A, 部门B, 约束准则B, 评分B, 部门C, 约束准则C, 评分C) \
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

    # 遍历 DataFrame 的每一行并插入数据
    for index, row in data.iterrows():
        val = tuple(row)
        cursor.execute(sql, val)

    connection.commit()
    cursor.close()


class Load:
    def __init__(self):
        self.connection = create_connection()

    def loading(self):

        # 读取Excel文件
        df = read_excel_file('example.xlsx')

        # 插入数据
        insert_data(self.connection, df)

        # 查询数据
        query = "SELECT * FROM strategies;"
        execute_query(self.connection, query)


if __name__ == "__main__":
    loader = Load()
    loader.loading()
