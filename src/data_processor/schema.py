import pandera.polars as pa
import polars as pl


class DataSchema(pa.DataFrameModel):
    # PassengerId: pl.Int64
    Pclass: pl.Int64
    Sex: pl.String
    Age: pl.Float64
    SibSp: pl.Int64
    Parch: pl.Int64
    Fare: pl.Float64
    Embarked: pl.String
