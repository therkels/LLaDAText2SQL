import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()


class DatabaseHelper():
    def __init__(self, db_name):
        self.__db_name = db_name
        # 1. Create the connection ONCE in the constructor
        self.__conn = sqlite3.connect(self.__db_name)

    def __enter__(self):
        # 2. __enter__ just returns a cursor from the
        #    *existing* connection
        return self.__conn.cursor()

    def __exit__(self, exc_type, exc_value, traceback):
        # 3. __exit__ just commits the transaction.
        #    DO NOT close the connection here.
        if self.__conn:
            self.__conn.commit()

    # Add a dedicated close method
    def close(self):
        """Closes the database connection."""
        if self.__conn:
            self.__conn.close()
            print(f"Database connection {self.__db_name} closed.")

    def insert_data(self, DDL_query):
        """
        Use executescript for multi-statement SQL strings.
        This is safer than manual splitting.
        """
        try:
            with self as cursor:
                # Use executescript to run multiple statements
                cursor.executescript(DDL_query)
    
        except sqlite3.Error as e:
            pass
            # print(f"An error occurred: {e}, skipping insertion")
            
    def fetch_data(self, DML_query):
        try:
            with self as cursor:
                cursor.execute(DML_query)
                # Get column names for the DataFrame
                cols = [column[0] for column in cursor.description]
                df = pd.DataFrame(cursor.fetchall(), columns=cols)
                return df, "success"
        except Exception as e:
            # print(f"An error occurred: {e}, returning empty DataFrame")
            # print(f"DEBUG: SQL that caused ERROR:\n{DML_query}")
            return pd.DataFrame(), "failure"
    
    def has_data(self, DML_query):
        try:
            with self as cursor:
                # print("RESULT")
                cursor.execute(DML_query)
                results = cursor.fetchall()
                # print("RESULT")
                # print(results)
                return len(results) > 0
        except sqlite3.Error as e:
            # print(f"An error occurred: {e}")
            return False



def example_usage():
    db_helper = DatabaseHelper(':memory:')

    # Create a table
    create_table__data_query = '''
      CREATE TABLE creative_ai (application_id INT, name TEXT, region TEXT, explainability_score FLOAT); INSERT INTO creative_ai (application_id, name, region, explainability_score) VALUES (1, 'ApplicationX', 'Europe', 0.87), (2, 'ApplicationY', 'North America', 0.91), (3, 'ApplicationZ', 'Europe', 0.84), (4, 'ApplicationAA', 'North America', 0.93), (5, 'ApplicationAB', 'Europe', 0.89);
    '''

    db_helper.insert_data(create_table__data_query)

    # Fetch data
    select_query = "SELECT AVG(explainability_score) FROM creative_ai WHERE region IN ('Europe', 'North America');"
    results = db_helper.fetch_data(select_query)

    # Convert to DataFrame for better visualization
    print(results)

if __name__ == "__main__":
    example_usage()
    # base_path = os.getenv("DB_DIR", None)
    # if not base_path:
    #     raise ValueError("Error: DB_DIR environment variable not set. Please set it in the .env file.")
    # new_db = DatabaseHelper(os.path.join(base_path, 'test_database.db'))
    # new_db.insert_data('CREATE TABLE test_table (id INT, value TEXT); INSERT INTO test_table (id, value) VALUES (1, "Sample1"), (2, "Sample2");')
    # df = new_db.fetch_data('SELECT * FROM test_table;')
    # print(df)
    # print(new_db.does_table_exist('fff'))
