"""
Executa o experimento completo: dados -> 3 MLPs -> comparacao -> modelo final.

    python src/treinar.py

Gera em models/ o preprocessador, o modelo escolhido e o pacote JSON da demo
web; em reports/ as figuras e o resumo de metricas usado no relatorio.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

import data as D
import modelo as M

RAIZ = Path(__file__).resolve().parents[1]
DIR_MODELOS = RAIZ / "models"
DIR_FIGS = RAIZ / "reports" / "figures"
DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_FIGS.mkdir(parents=True, exist_ok=True)

COR_ROUBO, COR_FURTO = "#C2410C", "#1D4ED8"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})


def main() -> dict:
    print("=" * 70)
    print("1. DADOS")
    print("=" * 70)
    df, rel = D.obter_dados()
    print(json.dumps({k: v for k, v in rel.items() if k != "mapa_de_colunas"},
                     indent=2, ensure_ascii=False, default=str))

    Xtr, Xval, Xte, ytr, yval, yte = M.separar(df)
    print(f"\ntreino={len(Xtr)}  validacao={len(Xval)}  teste={len(Xte)}")
    print(f"proporcao de ROUBO: treino={ytr.mean():.3f} "
          f"val={yval.mean():.3f} teste={yte.mean():.3f}")

    pre = M.construir_preprocessador()
    Xtr_t = pre.fit_transform(Xtr)          # ajuste APENAS no treino
    Xval_t = pre.transform(Xval)
    Xte_t = pre.transform(Xte)
    n_entradas = Xtr_t.shape[1]
    print(f"atributos apos one-hot + escala: {n_entradas}")

    base_val = M.metricas_baseline(yval)
    print(f"\nbaseline (sempre {base_val['classe_majoritaria']}): "
          f"acc={base_val['acuracia']:.4f}  f1={base_val['f1']:.4f}  "
          f"acc_bal={base_val['acuracia_balanceada']:.4f}  auc=0.5000")
    print("  -> com classes desbalanceadas, acuracia e F1 da classe majoritaria "
          "ja comecam altos;\n     a selecao usa ROC-AUC e o limiar e ajustado "
          "na validacao.")

    # --- figura: distribuicao das classes e dois atributos-chave ------------
    fig, eixos = plt.subplots(1, 3, figsize=(13, 3.6))
    cont = df["alvo"].map(M.CLASSES).value_counts()
    eixos[0].bar(cont.index, cont.values, color=[COR_FURTO, COR_ROUBO], width=0.55)
    eixos[0].set_title("Distribuicao das classes")
    for i, v in enumerate(cont.values):
        eixos[0].text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")

    tab = pd.crosstab(df["tipo_local"], df["alvo"], normalize="index")
    tab = tab.reindex(tab[1].sort_values().index).tail(8)
    eixos[1].barh(tab.index, tab[1], color=COR_ROUBO)
    eixos[1].axvline(df["alvo"].mean(), ls="--", c="gray", lw=1)
    eixos[1].set_title("Proporcao de roubo por tipo de local")
    eixos[1].set_xlim(0, 1)
    eixos[1].tick_params(labelsize=7)

    tab2 = pd.crosstab(df["periodo"], df["alvo"], normalize="index")
    eixos[2].bar(tab2.index, tab2[1], color=COR_ROUBO, width=0.55)
    eixos[2].axhline(df["alvo"].mean(), ls="--", c="gray", lw=1)
    eixos[2].set_title("Proporcao de roubo por periodo")
    eixos[2].set_ylim(0, 1)
    eixos[2].tick_params(labelsize=7, axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(DIR_FIGS / "01_exploracao.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("2. TREINAMENTO DAS ARQUITETURAS")
    print("=" * 70)
    resultados, historicos, modelos = {}, {}, {}
    for nome, cfg in M.ARQUITETURAS.items():
        cfg = {**cfg, "nome": nome.split(" ")[0]}
        mlp = M.construir_mlp(n_entradas, cfg)
        hist = M.treinar(mlp, Xtr_t, ytr, Xval_t, yval, cfg)
        p_val = mlp.predict(Xval_t, verbose=0).ravel()
        p_tr = mlp.predict(Xtr_t, verbose=0).ravel()
        limiar = M.escolher_limiar(yval, p_val)
        resultados[nome] = {
            "config": {k: v for k, v in cfg.items() if k != "descricao"},
            "descricao": cfg["descricao"],
            "parametros": int(mlp.count_params()),
            "epocas_efetivas": len(hist.history["loss"]),
            "limiar": limiar,
            "treino": M.metricas(ytr, p_tr, limiar),
            "val": M.metricas(yval, p_val, limiar),
            "val_limiar_05": M.metricas(yval, p_val, 0.5),
        }
        historicos[nome] = hist.history
        modelos[nome] = mlp
        r = resultados[nome]
        print(f"\n{nome}  ({r['parametros']} parametros, "
              f"{r['epocas_efetivas']} epocas)")
        print(f"  limiar escolhido na validacao: {limiar:.3f}")
        print(f"  treino: auc={r['treino']['roc_auc']:.4f} "
              f"acc_bal={r['treino']['acuracia_balanceada']:.4f}")
        print(f"  valid.: auc={r['val']['roc_auc']:.4f} "
              f"acc_bal={r['val']['acuracia_balanceada']:.4f} "
              f"recall_roubo={r['val']['recall']:.3f} "
              f"recall_furto={r['val']['especificidade']:.3f}")
        print(f"          (com limiar 0,5 o recall de furto seria "
              f"{r['val_limiar_05']['especificidade']:.3f})")

    tabela = M.tabela_comparativa(resultados)
    print("\n" + "=" * 70)
    print("3. COMPARACAO (conjunto de VALIDACAO)")
    print("=" * 70)
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    melhor = tabela.iloc[0]["Arquitetura"]
    print(f"\n-> arquitetura escolhida: {melhor}")

    # --- figura: curvas de treinamento das tres arquiteturas ----------------
    fig, eixos = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex="col")
    for j, nome in enumerate(M.ARQUITETURAS):
        h = historicos[nome]
        eixos[0, j].plot(h["loss"], label="treino", color=COR_FURTO)
        eixos[0, j].plot(h["val_loss"], label="validacao", color=COR_ROUBO)
        eixos[0, j].set_title(nome, fontsize=9)
        eixos[0, j].set_ylabel("perda" if j == 0 else "")
        eixos[1, j].plot(h["accuracy"], color=COR_FURTO)
        eixos[1, j].plot(h["val_accuracy"], color=COR_ROUBO)
        eixos[1, j].set_ylabel("acuracia" if j == 0 else "")
        eixos[1, j].set_xlabel("epoca")
    eixos[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Curvas de treinamento x validacao", y=1.0)
    fig.tight_layout()
    fig.savefig(DIR_FIGS / "02_curvas_todas.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("4. AVALIACAO FINAL NO CONJUNTO DE TESTE")
    print("=" * 70)
    mlp = modelos[melhor]
    limiar = resultados[melhor]["limiar"]
    p_te = mlp.predict(Xte_t, verbose=0).ravel()
    m_te = M.metricas(yte, p_te, limiar)
    base_te = M.metricas_baseline(yte)
    resultados[melhor]["teste"] = m_te
    print(f"  limiar de decisao (fixado na validacao): {limiar:.3f}\n")
    print(f"  {'metrica':22s} {'MLP':>8s} {'baseline':>9s}")
    for k in ["roc_auc", "acuracia_balanceada", "f1_macro", "acuracia",
              "recall", "especificidade", "precision", "f1"]:
        print(f"  {k:22s} {m_te[k]:8.4f} {base_te[k]:9.4f}")

    mc = confusion_matrix(yte, (p_te >= limiar).astype(int))

    # --- figura: curvas do modelo final + matriz de confusao + ROC ----------
    h = historicos[melhor]
    fig, eixos = plt.subplots(1, 4, figsize=(16, 3.7))
    eixos[0].plot(h["loss"], label="treino", color=COR_FURTO)
    eixos[0].plot(h["val_loss"], label="validacao", color=COR_ROUBO)
    eixos[0].set_title("Perda"); eixos[0].set_xlabel("epoca")
    eixos[0].legend(frameon=False, fontsize=8)
    eixos[1].plot(h["accuracy"], label="treino", color=COR_FURTO)
    eixos[1].plot(h["val_accuracy"], label="validacao", color=COR_ROUBO)
    eixos[1].set_title("Acuracia"); eixos[1].set_xlabel("epoca")
    eixos[1].legend(frameon=False, fontsize=8)

    eixos[2].imshow(mc, cmap="Blues")
    eixos[2].set_xticks([0, 1], ["FURTO", "ROUBO"])
    eixos[2].set_yticks([0, 1], ["FURTO", "ROUBO"])
    eixos[2].set_xlabel("previsto"); eixos[2].set_ylabel("real")
    eixos[2].set_title(f"Matriz de confusao (teste, limiar {limiar:.2f})")
    eixos[2].grid(False)
    for i in range(2):
        for j in range(2):
            eixos[2].text(j, i, f"{mc[i, j]:,}".replace(",", "."), ha="center",
                          va="center",
                          color="white" if mc[i, j] > mc.max() / 2 else "black",
                          fontweight="bold")

    fpr, tpr, _ = roc_curve(yte, p_te)
    eixos[3].plot(fpr, tpr, color=COR_ROUBO, lw=2,
                  label=f"AUC = {m_te['roc_auc']:.3f}")
    eixos[3].plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    # ponto de operacao no limiar escolhido
    eixos[3].scatter([1 - m_te["especificidade"]], [m_te["recall"]], s=45,
                     color=COR_FURTO, zorder=3, label=f"limiar {limiar:.2f}")
    eixos[3].set_xlabel("falso positivo"); eixos[3].set_ylabel("verdadeiro positivo")
    eixos[3].set_title("Curva ROC (teste)")
    eixos[3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Modelo selecionado: {melhor}", y=1.02)
    fig.tight_layout()
    fig.savefig(DIR_FIGS / "03_modelo_final.png", bbox_inches="tight")
    plt.close(fig)

    # --- persistencia --------------------------------------------------------
    joblib.dump(pre, DIR_MODELOS / "preprocessador.joblib")
    mlp.save(DIR_MODELOS / "mlp_final.keras")

    metadados = {
        "arquitetura": melhor,
        "origem_dos_dados": rel["origem"],
        "registros": int(rel["registros_finais"]),
        "n_entradas": int(n_entradas),
        "limiar": round(float(limiar), 4),
        "proporcao_roubo": round(float(rel["proporcao_roubo"]), 4),
        "metricas_teste": {k: round(float(v), 4) for k, v in m_te.items()},
        "baseline_teste": {k: (round(float(v), 4) if not isinstance(v, str) else v)
                           for k, v in base_te.items()},
    }
    M.exportar_para_web(mlp, pre, df, DIR_MODELOS / "modelo_web.json", metadados)

    resumo = {
        "relatorio_dados": {k: v for k, v in rel.items() if k != "mapa_de_colunas"},
        "tamanhos": {"treino": len(Xtr), "validacao": len(Xval), "teste": len(Xte)},
        "n_entradas": int(n_entradas),
        "resultados": resultados,
        "tabela": tabela.to_dict(orient="records"),
        "melhor": melhor,
        "limiar": float(limiar),
        "baseline_teste": base_te,
        "matriz_confusao_teste": mc.tolist(),
    }
    (RAIZ / "reports" / "resumo_experimentos.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tabela.to_csv(RAIZ / "reports" / "tabela_comparativa.csv", index=False)

    # regenera a aplicacao web com os pesos recem-treinados, para que a pagina
    # nunca fique servindo um modelo antigo
    import subprocess
    r = subprocess.run([sys.executable, str(RAIZ / "src" / "gerar_demo.py")],
                       capture_output=True, text=True)
    print("\n" + (r.stdout.strip() or r.stderr.strip()))

    print("artefatos salvos em models/, reports/ e app/index.html")
    return resumo


if __name__ == "__main__":
    main()
