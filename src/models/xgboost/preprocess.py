import polars as pl

import xgboost as xgb


def create_input_data_for_xgboost(
    df_x: pl.DataFrame, df_y: pl.DataFrame, reference=None
):
    if reference is not None:
        xgb_dataset = xgb.DMatrix(
            df_x.to_numpy(),
            label=df_y.to_numpy().ravel(),
            reference=reference,
        )
    else:
        xgb_dataset = xgb.DMatrix(
            df_x.to_numpy(),
            label=df_y.to_numpy().ravel(),
        )
    return xgb_dataset
