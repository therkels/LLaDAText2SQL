import pandas as pd
from collections import namedtuple
from typing import Iterator
from tqdm import tqdm
from database_interface import DatabaseHelper
import json
from datasets import load_dataset

SyntheticTextToSQLExample = namedtuple(
    "SyntheticTextToSQLExample",
    [
        "id",
        "domain",
        "domain_description",
        "sql_complexity",
        "sql_complexity_description",
        "sql_task_type",
        "sql_task_type_description",
        "sql_prompt",
        "sql_context",
        "sql",
        "sql_explanation",
    ],
)


def get_raw_data(file_path: str) -> pd.DataFrame:
    """Reads the raw synthetic text-to-SQL data from a CSV file.

    Args:
        file_path (str): The path to the dataset (parquet file).
    Returns:
        pd.DataFrame: A DataFrame containing the raw data.
    """
    data = pd.read_parquet(file_path)
    return data


def convert_data_to_namedtuples(data: pd.DataFrame) -> Iterator[SyntheticTextToSQLExample]:
    """Converts the DataFrame rows to a list of SyntheticTextToSQLExample namedtuples.

    Args:
        data (pd.DataFrame): The DataFrame containing the raw data.
    Returns:
        list[SyntheticTextToSQLExample]: A list of namedtuples representing each example.
    """
    for row in data.itertuples():
        yield SyntheticTextToSQLExample(
            id=row.id,
            domain=row.domain,
            domain_description=row.domain_description,
            sql_complexity=row.sql_complexity,
            sql_complexity_description=row.sql_complexity_description,
            sql_task_type=row.sql_task_type,
            sql_task_type_description=row.sql_task_type_description,
            sql_prompt=row.sql_prompt,
            sql_context=row.sql_context,
            sql=row.sql,
            sql_explanation=row.sql_explanation,
        )


def valid_columns():
    return [
        "id",
        "domain",
        "domain_description",
        "sql_complexity",
        "sql_complexity_description",
        "sql_task_type",
        "sql_task_type_description",
        "sql_prompt",
        "sql_context",
        "sql",
        "sql_explanation",
    ]


if __name__ == "__main__":
    file_path = "/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet"
    output_file = "/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/synthetic_text_to_sql/valid_test.json"
    data = get_raw_data(file_path)

    with open(output_file, 'w') as f: 
        for example in tqdm(convert_data_to_namedtuples(data), total=len(data)):
            db_obj = DatabaseHelper(":memory:")
            db_obj.insert_data(example.sql_context)
            # print(example.sql)
            if db_obj.has_data(example.sql):
                # print("adding example")
                
                # 1. Convert the namedtuple to a dictionary for correct JSON keys
                example_dict = example._asdict()
                
                # 2. Dump the dictionary as a JSON string
                json_string = json.dumps(example_dict)
                
                # 3. Write it as a single line (JSONL format) with no comma
                f.write(json_string + '\n')

    retrieved_dataset = load_dataset("json", data_files=output_file)
    
    print("\nSuccessfully retrieved dataset:")
    print(retrieved_dataset)
    
    print("\nFirst example from the *retrieved* dataset:")
    print(retrieved_dataset['train'][0])

    print("length of retrieved dataset:", len(retrieved_dataset['train']))
        
