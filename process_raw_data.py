import pandas as pd
from collections import namedtuple
from typing import Iterator
from tqdm import tqdm
from database_interface import DatabaseHelper
import json
from datasets import load_dataset, Dataset, load_from_disk

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
    for row in data:
        yield SyntheticTextToSQLExample(
            id=row['id'],
            domain=row['domain'],
            domain_description=row['domain_description'],
            sql_complexity=row['sql_complexity'],
            sql_complexity_description=row['sql_complexity_description'],
            sql_task_type=row['sql_task_type'],
            sql_task_type_description=row['sql_task_type_description'],
            sql_prompt=row['sql_prompt'],
            sql_context=row['sql_context'],
            sql=row['sql'],
            sql_explanation=row['sql_explanation'],
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

def sql_instance_generator():
    data = load_dataset("gretelai/synthetic_text_to_sql")["train"]
    for example in tqdm(convert_data_to_namedtuples(data), total=len(data)):
            db_obj = DatabaseHelper(":memory:")
            db_obj.insert_data(example.sql_context)
            # print(example.sql)
            if db_obj.has_data(example.sql):
                yield example._asdict()

if __name__ == "__main__":
    file_path = "/scratch/eecs595f25_class_root/eecs595f25_class/llada_data/synthetic_text_to_sql/synthetic_text_to_sql_test.snappy.parquet"
    output_folder = "./data"
    print("starting data processing")
    arrow_dataset = Dataset.from_generator(sql_instance_generator)
    print("splitting data")
    split_arrow_dataset = arrow_dataset.train_test_split(test_size=0.05, seed=68)
    split_arrow_dataset.save_to_disk(output_folder)

    print("loading data to verify")
    reloaded = load_from_disk(output_folder)
    print(reloaded)
    print("First train example:", reloaded['train'][0])
        
