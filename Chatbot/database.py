import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
# load_dotenv()

MYSQL_HOST = os.environ.get("MYSQL_HOST")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE")
MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_PORT = os.environ.get("MYSQL_PORT")

class MySQLDatabase:
    def __init__(self):
        self.username = MYSQL_USER
        self.password = MYSQL_PASSWORD
        self.database = MYSQL_DATABASE
        self.host = MYSQL_HOST
        self.connection = None
        self.reset()

    @classmethod
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                # port = MYSQL_PORT
            )
            if self.connection.is_connected():
                print("Successfully connected to the database")
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")

    @classmethod
    def select(self, table, columns='*', condition=None):
        try:
            self.connect()
            cursor = self.connection.cursor(dictionary=True)
            query = f"SELECT {columns} FROM {table}"
            if condition:
                query += f" WHERE {condition}"
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            self.close()
            return result
        except Error as e:
            print(f"Error: {e}")
            return None

    @classmethod
    def insert(self, table, data):
        try:
            self.connect()
            cursor = self.connection.cursor()
            placeholders = ', '.join(['%s'] * len(data))
            columns = ', '.join(data.keys())
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(data.values()))
            self.connection.commit()
            cursor.close()
            print("Insert successful")
            self.close()
        except Error as e:
            print(f"Error: {e}")

    @classmethod
    def update(self, table, updates, condition):
        try:
            self.connect()
            cursor = self.connection.cursor()
            update_clause = ', '.join([f"{k} = %s" for k in updates.keys()])
            query = f"UPDATE {table} SET {update_clause} WHERE {condition}"
            cursor.execute(query, list(updates.values()))
            self.connection.commit()
            cursor.close()
            print("Update successful")
            self.close()
        except Error as e:
            print(f"Error: {e}")

    @classmethod
    def delete(self, table, condition):
        try:
            self.connect()
            cursor = self.connection.cursor()
            query = f"DELETE FROM {table} WHERE {condition}"
            cursor.execute(query)
            self.connection.commit()
            cursor.close()
            print("Delete successful")
            self.close()
        except Error as e:
            print(f"Error: {e}")

    @classmethod
    def insert_into_database(self):
        data = {
            'question': self.question,
            'result_or_classification': self.result,
            'model_name': self.model_name,
            'input_prompt': self.input_prompt,
            'input_token': self.input_token,
            'output_token': self.output_token,
            'database_name': self.database_name,
            'cost': self.cost,
            'time_stamp': self.time_stamp,
            'time_taken_to_answer': self.time_taken_to_answer,
            'step': self.step,            
        }
        self.insert('llm_result', data)

    @classmethod
    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            print("Connection closed")

    @classmethod
    def reset(self):
        self.question = None
        self.result_or_classification = None
        self.model_name = None
        self.input_prompt = None
        self.input_token = None
        self.output_token = None
        self.database_name = None
        self.cost = None
        self.time_stamp = None
        self.time_taken_to_answer = None
        self.step = None