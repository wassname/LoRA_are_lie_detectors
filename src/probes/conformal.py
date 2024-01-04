
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from mapie.classification import MapieClassifier
from einops import rearrange
import torch

class MapieClassifier2(MapieClassifier):

    def fit(self, X_train, y_train, sample_weight=None, X_val=None, y_val=None, **kwargs):
        X_train = rearrange(X_train, 'b l h v -> b (l h v)')
        return super().fit(X_train.numpy(), y_train.numpy(), sample_weight, **kwargs)

    def predict_proba(self, X):
        X = rearrange(X, 'b l h v -> b (l h v)').numpy()
        return torch.from_numpy(super().predict(X)).float()
