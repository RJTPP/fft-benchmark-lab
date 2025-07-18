"""Utility functions for CSV related tasks."""

from pathlib import Path

import pandas as pd
import polars as pl


def read_csv_df(path: str | Path, use_pandas: bool = False) -> pl.DataFrame | pd.DataFrame:
    """
    Read a CSV file into a Polars or Pandas DataFrame.

    Parameters:
        path (str | Path): Path to the CSV file.
        use_pandas (bool): If True, return a Pandas DataFrame. Defaults to False (Polars).

    Returns:
        pl.DataFrame | pd.DataFrame: The loaded DataFrame in the requested format.

    Example:
        >>> df = read_csv_df("data.csv")         # Returns Polars DataFrame
        >>> df = read_csv_df("data.csv", True)   # Returns Pandas DataFrame
    """
    df = pl.read_csv(path)
    return df if not use_pandas else df.to_pandas()


def df_to_csv(df: pl.DataFrame | pd.DataFrame, path: str | Path) -> None:
    """
    Save a Pandas or Polars DataFrame to a CSV file using Polars.

    Parameters:
        df (pl.DataFrame | pd.DataFrame): The DataFrame to save.
            - If a Pandas DataFrame is provided, it will be converted to Polars.
        path (str | Path): Destination file path. Parent directories will be created if needed.
    
    Example:
        >>> df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        >>> df_to_csv(df, "out.csv")
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path))
    

def dict_to_csv(data: dict, path: str | Path, nested: bool | None = None) -> None:
    """
    Save a (possibly nested) dictionary to a CSV file using Polars.

    Parameters:
        data (dict): The input dictionary. Can be either:
            - Flat dict of lists (e.g., {'x': [1, 2], 'y': [3, 4]})
            - Nested dict of dicts (e.g., {'a': {'x': 1}, 'b': {'x': 2}})
        path (str | Path): The output CSV file path.
        nested (bool | None): 
            - If True: treat `data` as nested and flatten it to rows.
            - If False: treat `data` as flat. Raises an error if data is nested.
            - If None: automatically detect if the data is nested.
        id_col (str): Column name to use for outer dictionary keys when flattening nested data.

    Raises:
        ValueError: If nested data is detected but `nested` is False.

    Example:
        >>> data = {"a": {"x": 1, "y": 2}, "b": {"x": 3, "y": 4}}
        >>> dict_to_csv(data, "out.csv", nested=True)

    Output CSV:
        id,x,y
        a,1,2
        b,3,4
    """
    detected_nested = all(isinstance(v, dict) for v in data.values())
    if nested is None:
        nested = detected_nested
    
    if detected_nested != nested:
        if detected_nested:
            raise ValueError("Data appears to be nested, but 'nested' is set to False.")
        else:
            raise ValueError("Data does not appear to be nested, but 'nested' is set to True.")
    
    if nested:
        data = [{"id": outer_key, **inner_dict} for outer_key, inner_dict in data.items()]

    df = pl.DataFrame(data)
    polars_to_csv(df, path)
    

if __name__ == "__main__":
    path = "test/test/test.csv"
    path = Path(path)
    data = {
        "a": {"x": 1, "y": 2},
        "b": {"x": 1, "y": 2}
    }
    dict_to_csv(data, path, nested=None)