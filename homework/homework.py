# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Paso 1: Carga y limpieza
BASE = Path(__file__).resolve().parent.parent
TRAIN = BASE / "files" / "input" / "train_data.csv.zip"
TEST = BASE / "files" / "input" / "test_data.csv.zip"

df_train = pd.read_csv(TRAIN, compression="zip")
df_test = pd.read_csv(TEST, compression="zip")

df_train = df_train.rename(columns={"default payment next month": "default"})
df_test = df_test.rename(columns={"default payment next month": "default"})

if "ID" in df_train.columns:
    df_train = df_train.drop(columns=["ID"])
if "ID" in df_test.columns:
    df_test = df_test.drop(columns=["ID"])

if "EDUCATION" in df_train.columns:
    df_train["EDUCATION"] = df_train["EDUCATION"].where(df_train["EDUCATION"] <= 4, 4)
if "EDUCATION" in df_test.columns:
    df_test["EDUCATION"] = df_test["EDUCATION"].where(df_test["EDUCATION"] <= 4, 4)

df_train = df_train.dropna().reset_index(drop=True)
df_test = df_test.dropna().reset_index(drop=True)

# Paso 2: División en X e y
target_col = "default"
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col].astype(int)
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col].astype(int)

# Paso 3: Pipeline
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features),
        ("scaler", StandardScaler(with_mean=True, with_std=True), numerical_features),
    ],
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("pca", PCA()),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("classifier", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
    ]
)

# Paso 4: GridSearchCV
param_grid = {
    "pca__n_components": [20, X_train.shape[1] - 2],
    "feature_selection__k": [12],
    "classifier__kernel": ["rbf"],
    "classifier__gamma": [0.1],
}

model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

model.fit(X_train, y_train)

# Paso 5: Guardar modelo
models_dir = BASE / "files" / "models"
os.makedirs(models_dir, exist_ok=True)
compressed_model_path = models_dir / "model.pkl.gz"

with gzip.open(compressed_model_path, "wb") as file:
    pickle.dump(model, file)

# Paso 6: Métricas
best = model.best_estimator_
yhat_train = best.predict(X_train)
yhat_test = best.predict(X_test)


def metrics_block(y_true, y_pred, dataset):
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


m_train = metrics_block(y_train, yhat_train, "train")
m_test = metrics_block(y_test, yhat_test, "test")

output_dir = BASE / "files" / "output"
os.makedirs(output_dir, exist_ok=True)
metrics_path = output_dir / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(m_train) + "\n")
    fh.write(json.dumps(m_test) + "\n")

# Paso 7: Matrices de confusión
from sklearn.metrics import confusion_matrix


def cm_block(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


cm_train = cm_block(y_train, yhat_train, "train")
cm_test = cm_block(y_test, yhat_test, "test")

with open(metrics_path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(cm_train) + "\n")
    fh.write(json.dumps(cm_test) + "\n")