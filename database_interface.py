import sqlite3
import pandas as pd


class DatabaseHelper():
    def __init__(self, db_name):
        self.__db_name = db_name
        self__conn = None

    def __enter__(self):
        self.__conn = sqlite3.connect(self.__db_name)
        return self.__conn.cursor()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__conn:
            self.__conn.commit()
            self.__conn.close()

    def insert_data(self, DDL_query):
        with self as cursor:
            split_queries = DDL_query.strip().split(';')
            for query in split_queries:
                if query.strip():
                    cursor.execute(query)

    def fetch_data(self, DML_query):
        with self as cursor:
            cursor.execute(DML_query)
            df = pd.DataFrame(cursor.fetchall())
            return df


def example_usage():
    db_helper = DatabaseHelper('example.db')

    # Create a table
    create_table__data_query = '''
      CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');
    '''

    db_helper.insert_data(create_table__data_query)

    # Fetch data
    select_query = 'SELECT * FROM salesperson;'
    results = db_helper.fetch_data(select_query)

    # Convert to DataFrame for better visualization
    print(results)


if __name__ == "__main__":
    example_usage()
