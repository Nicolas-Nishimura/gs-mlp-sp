"""
Pre-processamento, arquiteturas de MLP e avaliacao.

Separado do notebook para que o mesmo codigo seja usado no treino e na
aplicacao web, garantindo que a transformacao dos atributos seja identica
nos dois lugares.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import COLUNAS_CATEGORICAS, COLUNAS_NUMERICAS, SEMENTE

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "models"
DIR_FIGS = RAIZ / "reports" / "figures"

CLASSES = {0: "FURTO", 1: "ROUBO"}


# --------------------------------------------------------------------------
# Pre-processamento
# --------------------------------------------------------------------------

def construir_preprocessador() -> ColumnTransformer:
    """Categoricas -> imputacao + one-hot; numericas -> imputacao + padronizacao.

    One-hot (e nao rotulo inteiro) porque bairro, periodo e tipo de local sao
    nominais: numerar as categorias criaria uma ordem inexistente que a rede
    tentaria interpretar como grandeza.
    StandardScaler porque a MLP e sensivel a escala: 'hora' (0-23) e
    'dias_ate_comunicacao' (0-365) dominariam a inicializacao dos pesos.
    """
    cat = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    return ColumnTransformer(
        [("cat", cat, COLUNAS_CATEGORICAS), ("num", num, COLUNAS_NUMERICAS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def separar(df: pd.DataFrame, semente: int = SEMENTE):
    """Divide em treino (70%), validacao (15%) e teste (15%), estratificado.

    O teste e separado PRIMEIRO e so e tocado na avaliacao final; a escolha da
    arquitetura usa exclusivamente a validacao.
    """
    X = df[COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS]
    y = df["alvo"].astype(int).to_numpy()

    X_resto, X_teste, y_resto, y_teste = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=semente)
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_resto, y_resto, test_size=0.1765, stratify=y_resto, random_state=semente)
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste


# --------------------------------------------------------------------------
# Arquiteturas
# --------------------------------------------------------------------------

ARQUITETURAS = {
    "A - rasa (32)": {
        "camadas": [32],
        "dropout": 0.0,
        "l2": 0.0,
        "batchnorm": False,
        "lr": 1e-3,
        "epocas": 120,
        "batch": 64,
        "descricao": (
            "Linha de base: uma unica camada oculta, sem regularizacao. "
            "Serve para medir quanto as arquiteturas maiores realmente agregam."
        ),
    },
    "B - profunda (64-32) + dropout": {
        "camadas": [64, 32],
        "dropout": 0.30,
        "l2": 0.0,
        "batchnorm": True,
        "lr": 1e-3,
        "epocas": 120,
        "batch": 64,
        "descricao": (
            "Duas camadas ocultas para compor interacoes entre atributos "
            "(ex.: periodo x tipo de local). Dropout 0,3 e BatchNormalization "
            "para estabilizar o treino e segurar o sobreajuste que a capacidade "
            "extra permitiria."
        ),
    },
    "C - larga (128-64-32) + L2": {
        "camadas": [128, 64, 32],
        "dropout": 0.40,
        "l2": 1e-4,
        "batchnorm": True,
        "lr": 5e-4,
        "epocas": 150,
        "batch": 128,
        "descricao": (
            "Tres camadas ocultas, dropout 0,4, penalizacao L2 e taxa de "
            "aprendizado reduzida. Testa se mais capacidade ainda traz ganho "
            "ou se o problema ja saturou."
        ),
    },
}


def construir_mlp(n_entradas: int, config: dict, semente: int = SEMENTE):
    """Monta a MLP em Keras conforme a configuracao da arquitetura.

    Ativacao oculta: ReLU - barata, nao satura para valores positivos e evita a
    dissipacao do gradiente que sigmoide/tanh causam em redes empilhadas.
    Saida: 1 neuronio com sigmoide -> probabilidade da classe ROUBO.
    Perda: binary_crossentropy, a verossimilhanca correta para saida sigmoide.
    Otimizador: Adam, que adapta a taxa por parametro e converge rapido em
    entradas esparsas vindas do one-hot.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    keras.utils.set_random_seed(semente)
    tf.config.experimental.enable_op_determinism()

    reg = regularizers.l2(config["l2"]) if config["l2"] else None

    modelo = keras.Sequential(name=config.get("nome", "mlp"))
    modelo.add(layers.Input(shape=(n_entradas,)))
    for n in config["camadas"]:
        modelo.add(layers.Dense(n, activation="relu", kernel_regularizer=reg))
        if config["batchnorm"]:
            modelo.add(layers.BatchNormalization())
        if config["dropout"]:
            modelo.add(layers.Dropout(config["dropout"]))
    modelo.add(layers.Dense(1, activation="sigmoid"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return modelo


def treinar(modelo, Xtr, ytr, Xval, yval, config, verbose=0):
    from tensorflow import keras

    paradas = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
    ]
    hist = modelo.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=config["epocas"],
        batch_size=config["batch"],
        callbacks=paradas,
        verbose=verbose,
    )
    return hist


# --------------------------------------------------------------------------
# Avaliacao
# --------------------------------------------------------------------------

def metricas(y_verd, y_prob, limiar: float = 0.5) -> dict:
    """Metricas de classificacao em um limiar de decisao.

    Alem das classicas, inclui duas que nao se deixam enganar pelo
    desbalanceamento entre as classes:
      - acuracia balanceada: media do acerto em ROUBO e do acerto em FURTO;
      - f1_macro: media do F1 das duas classes.
    'especificidade' e o recall da classe FURTO.
    """
    y_pred = (y_prob >= limiar).astype(int)
    return {
        "acuracia": accuracy_score(y_verd, y_pred),
        "acuracia_balanceada": balanced_accuracy_score(y_verd, y_pred),
        "precision": precision_score(y_verd, y_pred, zero_division=0),
        "recall": recall_score(y_verd, y_pred, zero_division=0),
        "especificidade": recall_score(y_verd, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_verd, y_pred, zero_division=0),
        "f1_macro": f1_score(y_verd, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_verd, y_prob),
    }


def metricas_baseline(y_verd) -> dict:
    """O que um classificador trivial (sempre a classe majoritaria) alcanca.

    Com classes desbalanceadas, esse piso pode ser alto em acuracia e em F1 da
    classe majoritaria. Qualquer metrica do modelo precisa ser lida contra ele.
    """
    maj = int(np.mean(y_verd) >= 0.5)
    prob = np.full(len(y_verd), float(maj))
    m = metricas(y_verd, prob, limiar=0.5)
    m["roc_auc"] = 0.5          # um classificador constante nao ordena nada
    m["classe_majoritaria"] = CLASSES[maj]
    return m


def escolher_limiar(y_verd, y_prob) -> float:
    """Limiar que maximiza a estatistica J de Youden (TPR - FPR) na validacao.

    Com 2/3 dos registros sendo roubo, o limiar padrao de 0,5 empurra quase
    tudo para a classe majoritaria. Maximizar J e o mesmo que maximizar a
    acuracia balanceada: da peso igual ao acerto em roubo e em furto.

    O limiar e escolhido SO na validacao; o teste continua intocado.
    Ajustar o limiar (e nao reponderar as classes no treino) mantem as
    probabilidades da rede calibradas a frequencia real, o que importa para
    a leitura do numero na aplicacao web.
    """
    fpr, tpr, limiares = roc_curve(y_verd, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(np.clip(limiares[i], 0.01, 0.99))


def tabela_comparativa(resultados: dict) -> pd.DataFrame:
    """Uma linha por arquitetura, ordenada pelo criterio de selecao: ROC-AUC.

    As quatro metricas pedidas no enunciado (acuracia, precision, recall e F1)
    estao todas aqui, calculadas no limiar escolhido de cada arquitetura. O
    ROC-AUC entra como criterio de ordenacao porque nao depende de limiar nem
    da proporcao entre as classes.
    """
    linhas = []
    for nome, r in resultados.items():
        v = r["val"]
        linhas.append({
            "Arquitetura": nome,
            "Parametros": r["parametros"],
            "Epocas": r["epocas_efetivas"],
            "Acuracia (val)": v["acuracia"],
            "Precision (val)": v["precision"],
            "Recall (val)": v["recall"],
            "F1 (val)": v["f1"],
            "ROC-AUC (val)": v["roc_auc"],
            "Limiar": r["limiar"],
        })
    return (pd.DataFrame(linhas)
            .sort_values("ROC-AUC (val)", ascending=False)
            .reset_index(drop=True))


def exportar_para_web(modelo, preproc, df_modelo, caminho: Path, metadados: dict):
    """Exporta pesos e vocabularios em JSON para a demo HTML (inferencia em JS)."""
    pesos = []
    for camada in modelo.layers:
        p = camada.get_weights()
        if not p:
            continue
        tipo = camada.__class__.__name__
        if tipo == "Dense":
            pesos.append({"tipo": "dense", "W": p[0].tolist(), "b": p[1].tolist(),
                          "ativacao": camada.activation.__name__})
        elif tipo == "BatchNormalization":
            gamma, beta, media, var = p
            pesos.append({"tipo": "batchnorm", "gamma": gamma.tolist(),
                          "beta": beta.tolist(), "media": media.tolist(),
                          "var": var.tolist(), "eps": float(camada.epsilon)})
    # Dropout nao tem peso e e inativo na inferencia: nao precisa ser exportado.

    ct = preproc
    onehot = ct.named_transformers_["cat"].named_steps["onehot"]
    escala = ct.named_transformers_["num"].named_steps["escala"]
    imp_cat = ct.named_transformers_["cat"].named_steps["imputacao"]
    imp_num = ct.named_transformers_["num"].named_steps["imputacao"]

    pacote = {
        "colunas_categoricas": COLUNAS_CATEGORICAS,
        "colunas_numericas": COLUNAS_NUMERICAS,
        "categorias": [list(map(str, c)) for c in onehot.categories_],
        "imput_cat": [str(v) for v in imp_cat.statistics_],
        "imput_num": [float(v) for v in imp_num.statistics_],
        "media": escala.mean_.tolist(),
        "escala": escala.scale_.tolist(),
        "pesos": pesos,
        "opcoes": {c: sorted(df_modelo[c].dropna().astype(str).unique().tolist())
                   for c in COLUNAS_CATEGORICAS},
        "metadados": metadados,
    }
    caminho.write_text(json.dumps(pacote), encoding="utf-8")
    return caminho
