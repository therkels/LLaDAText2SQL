import pandas as pd
from collections import namedtuple


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


def convert_data_to_namedtuples(data: pd.DataFrame) -> list[SyntheticTextToSQLExample]:
    """Converts the DataFrame rows to a list of SyntheticTextToSQLExample namedtuples.

    Args:
        data (pd.DataFrame): The DataFrame containing the raw data.
    Returns:
        list[SyntheticTextToSQLExample]: A list of namedtuples representing each example.
    """
    examples = []
    for _, row in data.iterrows():
        example = SyntheticTextToSQLExample(
            id=row["id"],
            domain=row["domain"],
            domain_description=row["domain_description"],
            sql_complexity=row["sql_complexity"],
            sql_complexity_description=row["sql_complexity_description"],
            sql_task_type=row["sql_task_type"],
            sql_task_type_description=row["sql_task_type_description"],
            sql_prompt=row["sql_prompt"],
            sql_context=row["sql_context"],
            sql=row["sql"],
            sql_explanation=row["sql_explanation"],
        )
        examples.append(example)
    return examples


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
    data = get_raw_data(file_path)

    examples = convert_data_to_namedtuples(data)

    # print the first example
    sample_entry = examples[0]

    print("*" * 50)
    print("Sample Entry")
    print("id:\n\t", sample_entry.id)
    print("domain:\n\t", sample_entry.domain)
    print("domain_description:\n\t", sample_entry.domain_description)
    print("sql_complexity:\n\t", sample_entry.sql_complexity)
    print("sql_complexity_description:\n\t", sample_entry.sql_complexity_description)
    print("sql_task_type:\n\t", sample_entry.sql_task_type)
    print("sql_task_type_description:\n\t", sample_entry.sql_task_type_description)
    print("sql_prompt:\n\t", sample_entry.sql_prompt)
    print("sql_context:\n\t", sample_entry.sql_context)
    print("sql:\n\t", sample_entry.sql)
    print("sql_explanation:\n\t", sample_entry.sql_explanation)
    print("*" * 50)
