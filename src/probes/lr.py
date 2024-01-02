import torch
import numpy as np
from einops import rearrange
from sklearn.preprocessing import StandardScaler, RobustScaler

class TorchRobustScaler(RobustScaler):

    def wrap(self, X, method: str):
        b, l, h, v = X.shape
        X = rearrange(X, "b l h v -> b (l h v)")
        X = getattr(super(), method)(X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(rearrange(X, "b (l h v) -> b l h v", l=l, h=h, v=v))
        return X

    def fit(self, X):
        return self.wrap(X, "fit")

    def transform(self, X):
        return self.wrap(X, "transform")

    def inverse_transform(self, X):
        return self.wrap(X, "inverse_transform")


from sklearn.linear_model import LogisticRegression
from einops import rearrange

class TorchLogisticRegression(LogisticRegression):

    def fit(self, X_train, y_train, sample_weight = None, X_val=None, y_val=None, **kwargs):
        X_train = rearrange(X_train, 'b l h v -> b (l h v)')
        return super().fit(X_train.numpy(), y_train.numpy(), sample_weight)
    
    def predict_proba(self, X):
        X = rearrange(X, 'b l h v -> b (l h v)').numpy()
        return torch.from_numpy(super().predict_proba(X)[:, 1])


from sklearn.dummy import DummyClassifier
from einops import rearrange

class TorchDummyClassifier(DummyClassifier):

    def fit(self, X_train, y_train, sample_weight = None, **kwargs):
        X_train = rearrange(X_train, 'b l h v -> b (l h v)')
        return super().fit(X_train.numpy(), y_train.numpy(), sample_weight)
    
    def predict_proba(self, X):
        X = rearrange(X, 'b l h v -> b (l h v)').numpy()
        return torch.from_numpy(super().predict_proba(X)[:, 1])
