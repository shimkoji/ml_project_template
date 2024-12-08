import pandera as pa
import polars as pl
from pandera.typing import DataFrame

from src.data_processor.schema import DataSchema


@pa.check_types
def replace_null_data_with_default(
    df: pl.DataFrame, replace_dict: dict
) -> DataFrame[DataSchema]:
    columns = [pl.col(col).fill_null(value) for col, value in replace_dict.items()]
    df_processed = df.with_columns(columns)
    return df_processed


# TODO
@pa.check_types
def replace_null_data_with_average(df: pl.DataFrame) -> DataFrame[DataSchema]:
    # df_processed = df.with_columns(
    #     pl.col("Age").fill_null(pl.col("Age").mean()),
    # )
    # return df_processed
    pass


def select_cols(df: pl.DataFrame, col_name_list: list[str]) -> pl.DataFrame:
    """指定された列のデータを抽出する(baseline用)"""
    df_selected = df.select(pl.col(col_name_list))
    return df_selected


def one_hot_encoding(df: pl.DataFrame, col_name_list: list[str]) -> pl.DataFrame:
    """指定された列をone-hotエンコーディングする"""
    df_encoded = df.with_columns(df.select(col_name_list).to_dummies())
    df_encoded = df_encoded.drop(col_name_list)
    return df_encoded


# TODO
def label_encoding(df: pl.DataFrame, col_name_list: list[str]) -> pl.DataFrame:
    """指定された列をラベルエンコーディングする"""
    pass


# TODO
def standard_scaling(df: pl.DataFrame, col_name_list: list[str]) -> pl.DataFrame:
    """指定された列を標準化する"""
    pass


# TODO
def minmax_scaling(df: pl.DataFrame, col_name_list: list[str]) -> pl.DataFrame:
    """指定された列を正規化する"""
    pass
