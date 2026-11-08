# Roubo ou furto? — classificação de ocorrências em São Paulo com uma MLP

Global Solution — **Aprendizado Profundo com Redes Neurais Convolucionais**

Uma rede neural *Multilayer Perceptron* que recebe as características de uma
ocorrência de subtração de celular registrada no município de São Paulo e estima se o
registro é mais compatível com **roubo** (art. 157 do Código Penal — subtração com
violência ou grave ameaça) ou com **furto** (art. 155 — subtração sem violência).

> **O que o modelo não faz.** Ele não prevê se um crime vai acontecer em determinado
> local ou horário. Recebe uma ocorrência que já existe — ou uma hipotética descrita
> no formulário — e classifica o **tipo** daquele registro.
>
> Saída de 72 % para "roubo" significa: *dado que uma subtração foi registrada com
> essas características, o modelo atribui 72 % de probabilidade a ela ter sido um
> roubo*. **Não** significa que há 72 % de chance de ocorrer um roubo naquele lugar.

---

## Aplicação web

**🔗 Link da aplicação:** _(preencher com a URL após publicar — veja "Publicar" abaixo)_

A aplicação recebe os oito campos que a equipe usou no treino (hora, período, dia da
semana, bairro, tipo de local, marca do aparelho, mês e dias até o registro do B.O.),
aciona a classificação pelo botão **Classificar** e mostra a classe estimada com as
duas probabilidades, além da leitura correta e da leitura incorreta do resultado.

Para rodar localmente, basta **abrir `app/index.html` no navegador** — arquivo único,
sem servidor e sem instalação. Os detalhes estão em *Como reproduzir › Rodar a
aplicação*.

A inferência em JavaScript foi conferida contra as previsões do Keras: diferença
máxima da ordem de **1e-7** (erro de ponto flutuante).

---

## Fonte dos dados

**Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP)** — Portal de
Estatísticas → Consultas → **Bases de Dados** → *Celulares subtraídos*

🔗 https://www.ssp.sp.gov.br/estatistica/consultas

### Procedimento de obtenção

1. Acessar o portal acima e abrir a seção **Bases de Dados**.
2. Baixar *Celulares subtraídos* para o ano escolhido (arquivo `.xlsx`).
3. Salvar o arquivo em `data/raw/` — pode colocar vários; são concatenados.
4. Executar `python src/treinar.py` (ou o notebook).

O carregador detecta as colunas automaticamente por lista de nomes candidatos, porque
o esquema do arquivo muda entre safras.

### Recorte aplicado

| | |
|---|---|
| **Geográfico** | Somente registros cujo **local do fato** é o município de São Paulo (campo `CIDADE`). Isso é diferente do local de registro do boletim. |
| **Rótulo** | `1 = ROUBO`, `0 = FURTO`, a partir do campo `RUBRICA`. Descartadas rubricas ambíguas e fora do recorte (latrocínio, receptação, estelionato, apropriação indébita, localização de objeto). |
| **Duplicidades** | Um mesmo B.O. pode listar vários celulares, gerando uma linha por aparelho. Fica **uma linha por boletim** (ano + número + delegacia). Os identificadores servem só para agrupar; nunca entram como atributo. |
| **Dimensionamento** | Amostra aleatória de até 20 000 registros após a limpeza. |

### ⚠️ Modo espelho

Se `data/raw/` estiver vazio, `src/data.py` gera uma **base sintética** com a mesma
estrutura de colunas e com as dependências estatísticas do fenômeno real. Serve para
validar o pipeline ponta a ponta e **não** é evidência empírica sobre criminalidade
em São Paulo.

A origem efetivamente usada é impressa no notebook, gravada em
`models/modelo_web.json` e **exibida em um aviso na própria aplicação web**. Com o
arquivo oficial em `data/raw/`, basta reexecutar o treino: o aviso some e a aplicação
passa a servir o modelo real, sem nenhuma outra alteração.

---

## Atributos de entrada

| Atributo | Tipo | Por que entrou |
|---|---|---|
| `periodo` | categórico | Roubo exige contato com a vítima, mais provável em horários de menor movimento e visibilidade. |
| `dia_semana` | categórico | Separa a dinâmica de dias úteis da de fim de semana. |
| `bairro` | categórico | Aproxima o contexto urbano do fato. 25 bairros mais frequentes + `OUTROS`. |
| `tipo_local` | categórico | O atributo mais informativo. Via pública favorece a abordagem direta; bar e comércio favorecem a subtração sem a vítima perceber. |
| `marca_objeto` | categórico | Aparelho de maior valor de revenda é alvo mais frequente de abordagem direta. |
| `hora` | numérico | Resolução mais fina que `periodo`. |
| `hora_sen`, `hora_cos` | numérico | A hora é circular: 23 h e 0 h são vizinhas, mas numericamente distariam 23 unidades. Seno e cosseno de 2πh/24 devolvem essa vizinhança à rede. |
| `mes` | numérico | Sazonalidade. |
| `fim_de_semana` | numérico (0/1) | Marca explicitamente o contraste sábado-domingo. |
| `dias_ate_comunicacao` | numérico | Quem é abordado com violência registra no mesmo dia; quem é furtado muitas vezes só nota a falta depois. |

### Excluídos deliberadamente

- **Vazamento de informação:** `RUBRICA`, `CONDUTA`, `DESDOBRAMENTO` — carregam a
  própria classificação. Mantidos na lista `COLUNAS_PROIBIDAS` de `src/data.py`; a
  `RUBRICA` só gera o rótulo e nunca chega ao conjunto de entrada.
- **Dados pessoais:** nome, RG, CPF, idade, sexo, nacionalidade, profissão.
- **Identificadores:** `NUM_BO`, `ANO_BO`, nome da delegacia.
- **Endereço exato:** `LOGRADOURO`, `NUMERO`, latitude e longitude — preterido em
  favor de `bairro`, que resume o contexto urbano com menos ruído de preenchimento.

---

## Arquiteturas testadas

| | Camadas ocultas | Regularização | LR | Batch | Pergunta |
|---|---|---|---|---|---|
| **A** | `32` | nenhuma | 1e-3 | 64 | Quanto uma rede mínima já resolve? |
| **B** | `64 → 32` | Dropout 0,3 + BatchNorm | 1e-3 | 64 | Profundidade compõe interações entre atributos? |
| **C** | `128 → 64 → 32` | Dropout 0,4 + L2 1e-4 + BatchNorm | 5e-4 | 128 | Mais capacidade ainda traz ganho? |

Todas com ReLU nas ocultas, **1 neurônio sigmoide** na saída,
perda `binary_crossentropy`, otimizador **Adam**, `EarlyStopping` (paciência 15,
`restore_best_weights`) e `ReduceLROnPlateau`.

Seleção pelo **ROC-AUC do conjunto de validação**: ele não depende do limiar de decisão
nem da proporção entre as classes. Isso importa porque 66,6% da base é roubo — um
classificador que responde sempre "roubo" já teria acurácia 0,666 e F1 0,800.

O **limiar de decisão** também é escolhido na validação (estatística J de Youden), em vez
de fixado em 0,5. Sem isso o modelo joga quase todo furto na classe roubo.

O conjunto de teste é separado antes de tudo e usado uma única vez, na avaliação final.

Os resultados da execução corrente ficam em `reports/resumo_experimentos.json` e
`reports/tabela_comparativa.csv`, e aparecem na própria aplicação web.

---

## Resultados da execução atual

Base oficial SSP-SP, *Celulares subtraídos* **2025**. De 383.635 registros do
arquivo, 216.129 têm o fato na capital; após rótulo, deduplicação por boletim e
limpeza temporal restam 78.868 ocorrências, das quais foi tirada uma
amostra de 20.000. **66,6% são roubo.**

Arquitetura escolhida: **B - profunda (64-32) + dropout**, limiar de decisão **0,67**.

| Métrica (teste) | MLP | Chutar sempre roubo |
|---|---|---|
| ROC-AUC | **0,745** | 0,500 |
| Acurácia balanceada | **0,661** | 0,500 |
| F1 macro | **0,640** | 0,400 |
| Acurácia | 0,654 | 0,666 |
| Precision (roubo) | **0,801** | 0,666 |
| Recall (roubo) | 0,640 | 1,000 |
| F1 (roubo) | 0,712 | 0,800 |

A leitura honesta: em **acurácia** e **F1 da classe roubo** o classificador trivial parece
competitivo, porque o desbalanceamento o favorece. Em ROC-AUC e acurácia balanceada, que
não se deixam enganar pela proporção das classes, o ganho real da rede aparece.

---

## Como reproduzir

```bash
git clone <url-deste-repositorio>
cd gs-mlp-sp

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Treinar

```bash
# coloque o .xlsx da SSP-SP em data/raw/ (opcional — sem ele roda a base espelho)
python src/treinar.py
```

Gera:

```
models/mlp_final.keras          modelo escolhido
models/preprocessador.joblib    ColumnTransformer ajustado no treino
models/modelo_web.json          pesos + vocabulários para a página estática
reports/figures/*.png           exploração, curvas, matriz de confusão, ROC
reports/resumo_experimentos.json
reports/tabela_comparativa.csv
```

### Notebook

```bash
jupyter lab notebooks/mlp_roubo_furto_sp.ipynb
```

O notebook é a entrega documentada: fonte dos dados, justificativa dos atributos,
pré-processamento, as três MLPs, comparação, curvas, matriz de confusão e análise de
overfitting.

### Rodar a aplicação

**Dê duplo clique em `app/index.html`.** Não precisa de Python, de servidor nem de
instalação — o arquivo já traz os pesos da rede embutidos e faz a inferência em
JavaScript no próprio navegador.

Pelo terminal, se preferir:

```bash
start app/index.html      # Windows
open app/index.html       # macOS
xdg-open app/index.html   # Linux
```

Só é necessário regerar a página depois de treinar o modelo de novo:

```bash
python src/gerar_demo.py   # reconstrói app/index.html com os pesos recém-treinados
```

### Publicar

A página estática é um arquivo único sem dependências — qualquer hospedagem serve:

- **GitHub Pages:** `docs/index.html` já está no repositório. Faça commit e ative
  Pages apontando para `/docs` na branch principal — o link sai pronto. Depois de
  cada novo treino, rode `python src/gerar_demo.py` e copie `app/index.html` por
  cima de `docs/index.html`.
- **Netlify / Vercel / Cloudflare Pages:** arraste a pasta `app/`.

Depois, preencha o link no topo deste README.

---

## Estrutura

```
gs-mlp-sp/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                     ← o .xlsx da SSP-SP vai aqui
│   └── processed/
├── src/
│   ├── data.py                  carga, recorte, rótulo, engenharia de atributos
│   ├── modelo.py                pré-processamento, arquiteturas, avaliação
│   ├── treinar.py               experimento completo
│   └── gerar_demo.py            monta app/index.html
├── notebooks/
│   └── mlp_roubo_furto_sp.ipynb
├── app/
│   ├── index.html               aplicação estática (gerada)
│   ├── template.html            fonte da página
│   └── inferencia.js            forward pass em JS
├── docs/
│   └── index.html               cópia publicada pelo GitHub Pages
├── models/
│   ├── mlp_final.keras          modelo treinado
│   ├── preprocessador.joblib    ColumnTransformer ajustado no treino
│   └── modelo_web.json          pesos e vocabulários para a página
└── reports/                     gerado por src/treinar.py
    ├── figures/                 exploração, curvas, matriz de confusão, ROC
    ├── resumo_experimentos.json
    └── tabela_comparativa.csv
    
```

---

## Limitações

1. **Viés de subnotificação.** A base só contém o que virou boletim. Furtos de
   aparelhos de baixo valor são sistematicamente menos notificados.
2. **A rubrica é uma classificação humana.** O rótulo vem do enquadramento feito no
   registro, que pode ser revisto na investigação.
3. **Qualidade de preenchimento.** Bairro e tipo de local têm preenchimento
   irregular; a imputação pela moda introduz uma suavização que não é neutra.
4. **Validade temporal e geográfica.** Treinado em um recorte de um ano na capital.
5. **Não é ferramenta de previsão nem de policiamento preditivo.** A saída é a
   compatibilidade de um registro com cada tipo — não é risco, e não deve orientar
   alocação de efetivo.

---

## Equipe

|       Nome      |   RM   |
|-----------------|--------|
| Lincoln Pereira | 567284 |
| Miguel Bezzerra | 566763 |
| Nicolas Sakaue  | 567752 |