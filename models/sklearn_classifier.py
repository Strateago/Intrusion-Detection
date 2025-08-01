import typing

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Mapping of available models
MODELS_FACTORY = {
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier
}

class SklearnClassifier():
    def __init__(self, model_hyperparams: typing.Dict):
        super(SklearnClassifier, self).__init__()
        self._model_name = model_hyperparams["model_name"]
        self._model_params = model_hyperparams["model_params"]

        if self._model_name not in MODELS_FACTORY:
            raise KeyError(f"Selected model '{self._model_name}' is NOT available!")

        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('clf', MODELS_FACTORY[self._model_name](**self._model_params))
        ]

        self._model = Pipeline(pipeline_steps)

    def reset(self):
        self._model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MODELS_FACTORY[self._model_name](**self._model_params))
        ])

    def train(self, X_data, y_data):
        self._model.fit(X_data, y_data)

    def predict(self, X_data):
        return self._model.predict(X_data)

    def predict_proba(self, X_data):
        clf = self._model.named_steps['clf']
        if hasattr(clf, "predict_proba"):
            return self._model.predict_proba(X_data)
        else:
            raise AttributeError(f"{self._model_name} does not support predict_proba.")
