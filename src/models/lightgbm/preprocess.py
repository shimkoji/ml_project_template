import lightgbm as lgb
import polars as pl

# split the dataset into the training set and validation set
def create_input_data_for_lightgbm(df_x: pl.DataFrame, df_y: pl.DataFrame, reference=None):
    if reference is not None:
        lgb_dataset = lgb.Dataset(df_x.to_numpy(), label=df_y.to_numpy().ravel(), reference=reference,)
    else:
        lgb_dataset = lgb.Dataset(df_x.to_numpy(), label=df_y.to_numpy().ravel())
    return lgb_dataset
