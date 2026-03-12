import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):

        X_df = pd.DataFrame(X)

        corr_matrix = X_df.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop_ = [
            column for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        return self

    def transform(self, X):

        X_df = pd.DataFrame(X)

        return X_df.drop(columns=self.to_drop_, errors="ignore").values