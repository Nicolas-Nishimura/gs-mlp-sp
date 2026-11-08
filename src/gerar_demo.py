"""
Monta app/index.html: template + inferencia.js + pesos do modelo, em um unico
arquivo autossuficiente (abre com duplo clique, sem servidor).

    python src/gerar_demo.py
"""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

template = (RAIZ / "app" / "template.html").read_text(encoding="utf-8")
inferencia = (RAIZ / "app" / "inferencia.js").read_text(encoding="utf-8")
pacote = (RAIZ / "models" / "modelo_web.json").read_text(encoding="utf-8")
tabela = json.loads((RAIZ / "reports" / "resumo_experimentos.json")
                    .read_text(encoding="utf-8"))["tabela"]

html = (template
        .replace("/*__INFERENCIA__*/", inferencia)
        .replace("/*__PACOTE__*/", pacote)
        .replace("/*__COMPARATIVO__*/", json.dumps(tabela, ensure_ascii=False)))

destino = RAIZ / "app" / "index.html"
destino.write_text(html, encoding="utf-8")
print(f"{destino}  ({destino.stat().st_size / 1024:.0f} KB)")
