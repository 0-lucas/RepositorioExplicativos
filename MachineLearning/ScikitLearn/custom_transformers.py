from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# Transformador sem herança - Possível de integrar ao Pipeline. Sem flexibilidade.
class Square:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X ** 2


# Usando Scikit Learn - Possível de integrar ao Pipeline. Boa prática e melhor deploy.
class CombineAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, column="Dorm",
                 column_name="Notebooks in Dorm", to_integer=True):

        self.column = column
        self.column_name = column_name
        self.to_integer = to_integer

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.groupby(self.column).sum(numeric_only=True)
        X_rename = X_new.rename(columns={X_new.columns[0]: self.column_name})  # Renomeando para melhor visualização.
        X_merged = X.join(X_rename, on=self.column)

        if self.to_integer:
            X_rounded = X_merged.astype({self.column_name: "Int8"})
            return X_rounded

        return X_merged


# Criando uma funcão usando apenas pandas.
def add_combined_column(dataframe, column, new_column_name="new_column"):
    dataframe_grouped = dataframe.groupby(column).sum(numeric_only=True)

    column_to_rename = dataframe_grouped.columns[0]
    dataframe_renamed = dataframe_grouped.rename(columns={column_to_rename: new_column_name})

    dataframe_merged = dataframe.join(dataframe_renamed, on="Dorm")
    return dataframe_merged
