"""
Camada de dados do projeto: carregamento, recorte, rotulagem e engenharia de atributos.

Fonte oficial (a ser baixada pela equipe):
    Secretaria da Seguranca Publica do Estado de Sao Paulo (SSP-SP)
    Portal de Estatisticas - Consultas > Bases de Dados > "Celulares subtraidos"
    https://www.ssp.sp.gov.br/estatistica/consultas

O modulo opera em dois modos:

1) MODO OFICIAL  -> ha um arquivo .xlsx/.csv da SSP-SP em data/raw/.
   `carregar_base_ssp()` detecta o arquivo, normaliza os nomes de coluna e
   mapeia automaticamente os campos usados pelo projeto. O esquema da SSP-SP
   muda de ano para ano, entao o mapeamento e feito por lista de candidatos.

2) MODO ESPELHO  -> nao ha arquivo em data/raw/.
   `gerar_base_espelho()` sintetiza uma base com a MESMA estrutura de colunas
   e com dependencias estatisticas compativeis com o fenomeno real
   (roubo concentrado em via publica / periodo noturno / comunicacao imediata;
   furto concentrado em locais fechados / periodo diurno / comunicacao tardia).
   Serve para validar o pipeline ponta a ponta. NAO deve ser usada como
   evidencia empirica sobre criminalidade em Sao Paulo.

O modo ativo fica registrado em `ORIGEM_DOS_DADOS` e e impresso no notebook.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"

SEMENTE = 42

# --------------------------------------------------------------------------
# Esquema alvo do projeto
# --------------------------------------------------------------------------

# Campos que o projeto consome. Para cada um, lista de nomes candidatos que a
# SSP-SP ja utilizou em diferentes safras do arquivo.
CANDIDATOS = {
    "data_ocorrencia": ["dataocorrencia", "data_ocorrencia", "dataocorrenciabo", "data_fato"],
    "hora_ocorrencia": ["horaocorrencia", "hora_ocorrencia", "hora_fato"],
    "periodo": ["peridoocorrencia", "periodoocorrencia", "periodo_ocorrencia", "periodo"],
    "data_comunicacao": ["datacomunicacao", "data_comunicacao", "dataregistro", "bo_emitido"],
    "cidade": ["cidade", "municipio", "cidade_ocorrencia", "nome_municipio"],
    "bairro": ["bairro", "bairro_ocorrencia", "nome_bairro"],
    "tipo_local": ["descr_tipolocal", "descrtipolocal", "tipo_local", "descricaolocal", "local"],
    "marca_objeto": ["descr_marca_celular", "marca_celular", "descr_marca", "marca", "descr_marca_objeto"],
    "rubrica": ["rubrica", "natureza", "natureza_apurada", "especie", "descricao_rubrica"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "long", "lng"],
}

# Colunas que NUNCA entram como atributo de entrada: revelam o rotulo
# (vazamento) ou carregam dado pessoal / identificador de boletim.
COLUNAS_PROIBIDAS = {
    "rubrica", "conduta", "desdobramento", "especie", "natureza", "natureza_apurada",
    "num_bo", "numero_boletim", "ano_bo", "bo_iniciado", "bo_emitido", "delegacia_nome",
    "delegacia_circunscricao", "nome", "nomepessoa", "nome_pessoa", "rg", "cpf",
    "idade", "sexo", "datanascimento", "nacionalidade", "naturalidade", "profissao",
    "logradouro", "numero", "complemento", "placa", "imei", "serie",
}

# Identificadores do boletim. Usados SOMENTE para reconhecer que varias linhas
# descrevem a mesma ocorrencia (um B.O. pode listar varios objetos subtraidos).
# Continuam proibidos como atributo de entrada: ver COLUNAS_PROIBIDAS.
CANDIDATOS_ID = {
    "num_bo": ["num_bo", "numero_bo", "numero_boletim", "nr_bo", "num_boletim"],
    "ano_bo": ["ano_bo", "ano_boletim", "ano_referencia"],
    "delegacia": ["delegacia_nome", "nome_delegacia", "delegacia", "nome_departamento",
                  "nome_delegacia_circunscricao"],
}

ORIGEM_DOS_DADOS: str | None = None


# --------------------------------------------------------------------------
# Utilitarios
# --------------------------------------------------------------------------

def _normaliza(texto: str) -> str:
    """minusculas, sem acento, sem pontuacao -> comparacao robusta de nomes."""
    if texto is None:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t


def _normaliza_valor(texto) -> str:
    if pd.isna(texto):
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).upper().strip()


def _mapear(df: pd.DataFrame, candidatos_por_alvo: dict) -> dict[str, str]:
    reais = {_normaliza(c): c for c in df.columns}
    mapa: dict[str, str] = {}
    for alvo, candidatos in candidatos_por_alvo.items():
        for cand in candidatos:
            if cand in reais:
                mapa[alvo] = reais[cand]
                break
        else:  # fallback: casamento por prefixo
            for norm, original in reais.items():
                if any(norm.startswith(c[:12]) for c in candidatos):
                    mapa[alvo] = original
                    break
    return mapa


def _mapear_colunas(df: pd.DataFrame) -> dict[str, str]:
    """Liga cada campo do esquema alvo a uma coluna real do arquivo."""
    return _mapear(df, CANDIDATOS)


# --------------------------------------------------------------------------
# 1) Modo oficial
# --------------------------------------------------------------------------

LINHAS_INSPECAO = 40          # ate onde procurar a linha de cabecalho

# conjunto achatado de todos os nomes de coluna que o projeto reconhece
_TODOS_CANDIDATOS = {c for lista in CANDIDATOS.values() for c in lista}


def _pontuar_cabecalho(celulas) -> int:
    """Quantos nomes reconhecidos aparecem nesta linha."""
    normalizadas = {_normaliza(c) for c in celulas if c is not None and not pd.isna(c)}
    pontos = 0
    for cand in _TODOS_CANDIDATOS:
        if cand in normalizadas or any(nz.startswith(cand[:12]) for nz in normalizadas):
            pontos += 1
    return pontos


def _achar_linha_cabecalho(bruto: pd.DataFrame) -> int | None:
    """Indice da linha que funciona como cabecalho, ou None se nao houver.

    O arquivo da SSP-SP costuma trazer linhas de titulo ou de metodologia antes
    da tabela, e as vezes uma aba inteira so com texto explicativo. Em vez de
    assumir que o cabecalho esta na primeira linha, procuramos a linha que mais
    se parece com um cabecalho conhecido.
    """
    melhor_linha, melhor_pontos = None, 0
    for i in range(min(LINHAS_INSPECAO, len(bruto))):
        pontos = _pontuar_cabecalho(bruto.iloc[i].tolist())
        if pontos > melhor_pontos:
            melhor_linha, melhor_pontos = i, pontos
    # exige pelo menos tres colunas reconhecidas para nao confundir uma linha
    # de dados com um cabecalho
    return melhor_linha if melhor_pontos >= 3 else None


def _ler_excel(caminho: Path) -> pd.DataFrame:
    """Varre todas as abas e devolve aquela que contem a tabela de ocorrencias."""
    xls = pd.ExcelFile(caminho)
    melhor = None                      # (pontos, n_linhas, nome_aba, df)
    descartadas = []

    for aba in xls.sheet_names:
        bruto = pd.read_excel(xls, sheet_name=aba, dtype=str, header=None,
                              nrows=LINHAS_INSPECAO)
        if bruto.empty:
            descartadas.append(f"{aba} (vazia)")
            continue

        linha = _achar_linha_cabecalho(bruto)
        if linha is None:
            descartadas.append(f"{aba} (sem cabecalho reconhecivel)")
            continue

        df = pd.read_excel(xls, sheet_name=aba, dtype=str, header=linha)
        df = df.dropna(axis=1, how="all")
        # remove colunas sem nome, geradas por celulas mescladas
        df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
        df = df.dropna(how="all")

        pontos = len(_mapear_colunas(df))
        if melhor is None or (pontos, len(df)) > (melhor[0], melhor[1]):
            melhor = (pontos, len(df), aba, df)

    if melhor is None:
        raise ValueError(
            f"nenhuma aba com tabela de ocorrencias. Abas vistas: {descartadas}")

    pontos, n_linhas, aba, df = melhor
    if len(xls.sheet_names) > 1:
        print(f"    aba usada: '{aba}' ({n_linhas} linhas, "
              f"{pontos} campos reconhecidos)")
    return df


def _ler_csv(caminho: Path) -> pd.DataFrame:
    for sep in [";", ",", "\t", "|"]:
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                bruto = pd.read_csv(caminho, sep=sep, dtype=str, encoding=enc,
                                    header=None, nrows=LINHAS_INSPECAO,
                                    on_bad_lines="skip", low_memory=False)
            except Exception:
                continue
            if bruto.shape[1] <= 3:
                continue
            linha = _achar_linha_cabecalho(bruto)
            if linha is None:
                continue
            return pd.read_csv(caminho, sep=sep, dtype=str, encoding=enc,
                               skiprows=linha, on_bad_lines="skip",
                               low_memory=False)
    raise ValueError(f"nao consegui identificar o cabecalho em {caminho.name}")


def _ler_arquivo(caminho: Path) -> pd.DataFrame:
    if caminho.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        return _ler_excel(caminho)
    return _ler_csv(caminho)


def carregar_base_ssp(dir_raw: Path = DIR_RAW) -> pd.DataFrame | None:
    """Le todos os arquivos de data/raw/ e devolve um unico DataFrame cru."""
    if not dir_raw.exists():
        return None
    arquivos = [p for p in sorted(dir_raw.iterdir())
                if p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv", ".txt"}
                and not p.name.startswith("~$")]
    if not arquivos:
        return None

    partes = []
    for arq in arquivos:
        try:
            df = _ler_arquivo(arq)
            print(f"  lido: {arq.name}  ({len(df)} linhas, {df.shape[1]} colunas)")
            partes.append(df)
        except Exception as e:
            print(f"  ignorado: {arq.name} ({e})")
    if not partes:
        return None
    # arquivos de meses diferentes podem ter colunas em ordem distinta;
    # concat alinha pelo nome e preenche o que faltar
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------
# 2) Modo espelho (base sintetica de validacao do pipeline)
# --------------------------------------------------------------------------

BAIRROS = [
    "SE", "REPUBLICA", "SANTA CECILIA", "BELA VISTA", "CONSOLACAO", "PINHEIROS",
    "ITAIM BIBI", "VILA MARIANA", "MOEMA", "SANTANA", "TATUAPE", "MOOCA",
    "LAPA", "BUTANTA", "CAMPO BELO", "SAUDE", "IPIRANGA", "PENHA", "SANTO AMARO",
    "BRASILANDIA", "CAPAO REDONDO", "GRAJAU", "ITAQUERA", "SAO MATEUS", "JABAQUARA",
]
# peso relativo de volume por bairro e tendencia local a roubo (logito)
BAIRRO_PESO = np.array([9, 8, 6, 5, 5, 6, 4, 4, 3, 4, 4, 3, 3, 3, 2, 3, 3, 4, 4, 3, 4, 3, 5, 4, 3], float)
BAIRRO_PESO /= BAIRRO_PESO.sum()
BAIRRO_EFEITO = np.array([
    0.45, 0.50, 0.35, 0.20, 0.15, -0.35, -0.45, -0.25, -0.50, -0.05, -0.10, 0.05,
    0.00, -0.15, -0.30, -0.10, 0.10, 0.30, -0.20, 0.55, 0.60, 0.50, 0.40, 0.45, 0.05,
])

TIPOS_LOCAL = [
    "VIA PUBLICA", "ESTABELECIMENTO COMERCIAL", "TRANSPORTE PUBLICO",
    "INTERIOR DE VEICULO", "BAR / RESTAURANTE", "RESIDENCIA",
    "ESTACAO / TERMINAL", "ESTABELECIMENTO DE ENSINO", "OUTROS",
]
LOCAL_PESO = np.array([0.50, 0.13, 0.09, 0.05, 0.07, 0.06, 0.05, 0.03, 0.02])
LOCAL_EFEITO = np.array([1.05, -1.20, -0.55, -0.70, -1.35, 0.15, -0.85, -1.05, -0.15])

MARCAS = ["SAMSUNG", "APPLE", "MOTOROLA", "XIAOMI", "LG", "ASUS", "NAO INFORMADO", "OUTRAS"]
MARCA_PESO = np.array([0.36, 0.24, 0.17, 0.09, 0.03, 0.02, 0.06, 0.03])
MARCA_EFEITO = np.array([-0.05, 0.38, -0.10, 0.02, -0.12, -0.08, 0.05, 0.00])

PERIODOS = ["MADRUGADA", "PELA MANHA", "A TARDE", "A NOITE"]
PERIODO_EFEITO = np.array([0.75, -0.55, -0.30, 0.55])


def gerar_base_espelho(n: int = 18000, semente: int = SEMENTE,
                       intercepto: float = -0.35) -> pd.DataFrame:
    """Base sintetica com o esquema da SSP-SP. Uso: validar o pipeline."""
    rng = np.random.default_rng(semente)

    i_bairro = rng.choice(len(BAIRROS), size=n, p=BAIRRO_PESO)
    i_local = rng.choice(len(TIPOS_LOCAL), size=n, p=LOCAL_PESO)
    i_marca = rng.choice(len(MARCAS), size=n, p=MARCA_PESO)

    # hora depende do tipo de local (comercio de dia, bar de noite, etc.)
    centro_hora = np.array([16.0, 14.5, 12.0, 15.0, 22.5, 19.0, 13.5, 11.0, 15.0])[i_local]
    disp_hora = np.array([5.5, 3.2, 4.0, 4.5, 3.0, 4.5, 3.5, 2.5, 5.0])[i_local]
    hora = np.mod(rng.normal(centro_hora, disp_hora), 24)
    h_int = hora.astype(int)

    i_periodo = np.select(
        [h_int < 6, h_int < 12, h_int < 18],
        [0, 1, 2], default=3,
    )

    data_base = pd.Timestamp("2025-01-01")
    dia_do_ano = rng.integers(0, 365, size=n)
    data_oc = data_base + pd.to_timedelta(dia_do_ano, unit="D")
    dia_semana = data_oc.dayofweek.to_numpy()

    # logito do evento "roubo"
    logito = (
        intercepto
        + BAIRRO_EFEITO[i_bairro]
        + LOCAL_EFEITO[i_local]
        + MARCA_EFEITO[i_marca]
        + PERIODO_EFEITO[i_periodo]
        + 0.12 * (dia_semana >= 5)
        + rng.normal(0, 0.85, size=n)          # heterogeneidade nao observada
    )
    p_roubo = 1 / (1 + np.exp(-logito))
    eh_roubo = rng.random(n) < p_roubo

    # atraso ate a comunicacao: vitima de roubo registra quase sempre no mesmo dia
    atraso = np.where(
        eh_roubo,
        rng.choice([0, 0, 0, 1, 1, 2, 3], size=n),
        rng.choice([0, 1, 1, 2, 3, 4, 6, 9, 14, 21], size=n),
    )
    data_com = data_oc + pd.to_timedelta(atraso, unit="D")

    rubrica = np.where(
        eh_roubo,
        "Roubo (art. 157) - OUTROS",
        "Furto (art. 155) - OUTROS",
    )

    # ruido de preenchimento, como no arquivo real
    bairro = np.array(BAIRROS)[i_bairro].astype(object)
    bairro[rng.random(n) < 0.03] = np.nan
    tipo_local = np.array(TIPOS_LOCAL)[i_local].astype(object)
    tipo_local[rng.random(n) < 0.05] = np.nan

    df = pd.DataFrame({
        "ANO_BO": 2025,
        "NUM_BO": rng.permutation(n) + 100000,          # unico por ocorrencia
        "DATAOCORRENCIA": data_oc.strftime("%d/%m/%Y"),
        "HORAOCORRENCIA": [f"{h:02d}:{m:02d}" for h, m in zip(h_int, rng.integers(0, 60, n))],
        "PERIDOOCORRENCIA": np.array(PERIODOS)[i_periodo],
        "DATACOMUNICACAO": data_com.strftime("%d/%m/%Y"),
        "CIDADE": "S.PAULO",
        "BAIRRO": bairro,
        "DESCR_TIPOLOCAL": tipo_local,
        "DESCR_MARCA_CELULAR": np.array(MARCAS)[i_marca],
        "RUBRICA": rubrica,
        "LATITUDE": np.round(rng.normal(-23.55, 0.09, n), 6),
        "LONGITUDE": np.round(rng.normal(-46.63, 0.10, n), 6),
    })
    return df


# --------------------------------------------------------------------------
# 3) Recorte, rotulagem e engenharia de atributos
# --------------------------------------------------------------------------

def _parse_data(serie: pd.Series) -> pd.Series:
    """Datas chegam em formatos diferentes conforme a safra do arquivo.

    O .xlsx de 2025 traz datas do Excel, que o pandas le como texto ISO
    (2025-03-14 00:00:00); arquivos antigos e CSVs trazem 14/03/2025.
    Tenta os formatos explicitos primeiro, para nao inverter dia e mes.
    """
    s = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M:%S"]:
        faltando = s.isna()
        if not faltando.any():
            break
        s = s.where(~faltando, pd.to_datetime(serie[faltando], format=fmt, errors="coerce"))
    return s


def _parse_hora(serie: pd.Series) -> pd.Series:
    """Extrai a hora cheia de formatos variados (HH:MM, HH:MM:SS, HHMM, datetime)."""
    txt = serie.astype(str).str.strip()
    h = txt.str.extract(r"^(\d{1,2})\s*[:h]", expand=False)
    sem = h.isna()
    if sem.any():
        h2 = txt[sem].str.extract(r"^(\d{2})\d{2}$", expand=False)
        h = h.where(~sem, h2)
    h = pd.to_numeric(h, errors="coerce")
    return h.where(h.between(0, 23))


def _rotular(rubrica: pd.Series) -> pd.Series:
    """1 = roubo, 0 = furto, NaN = fora do escopo (descartado)."""
    r = rubrica.map(_normaliza_valor)
    eh_roubo = r.str.contains("ROUBO", na=False)
    eh_furto = r.str.contains("FURTO", na=False)
    # "latrocinio" e roubo seguido de morte: fora do recorte de celulares
    fora = r.str.contains("LATROCINIO|RECEPTACAO|ESTELIONATO|APROPRIACAO|ENCONTRO|LOCALIZACAO",
                          na=False, regex=True)
    rotulo = pd.Series(np.nan, index=rubrica.index, dtype="float64")
    rotulo[eh_furto & ~eh_roubo & ~fora] = 0.0
    rotulo[eh_roubo & ~eh_furto & ~fora] = 1.0
    return rotulo


def _agrupar_raros(serie: pd.Series, top: int, rotulo_outros: str = "OUTROS") -> pd.Series:
    freq = serie.value_counts()
    manter = set(freq.head(top).index)
    return serie.where(serie.isin(manter), rotulo_outros)


def preparar(
    df_cru: pd.DataFrame,
    top_bairros: int = 25,
    top_marcas: int = 8,
    amostra_max: int | None = 20000,
    semente: int = SEMENTE,
) -> tuple[pd.DataFrame, dict]:
    """Aplica recorte geografico, rotulagem e engenharia de atributos.

    Retorna (df_modelo, relatorio) onde df_modelo tem exatamente as colunas de
    entrada + a coluna 'alvo'.
    """
    rel: dict = {"registros_iniciais": len(df_cru)}
    mapa = _mapear_colunas(df_cru)
    faltando = [c for c in ["data_ocorrencia", "rubrica"] if c not in mapa]
    if faltando:
        raise KeyError(
            f"Colunas essenciais nao encontradas no arquivo: {faltando}. "
            f"Colunas disponiveis: {list(df_cru.columns)[:30]}"
        )
    rel["mapa_de_colunas"] = mapa

    df = pd.DataFrame(index=df_cru.index)

    # --- recorte geografico: municipio de Sao Paulo (local do FATO) ----------
    if "cidade" in mapa:
        cidade = df_cru[mapa["cidade"]].map(_normaliza_valor)
        padrao_capital = r"^(S\.?\s*PAULO|SAO PAULO|SAOPAULO|SP CAPITAL)$"
        mascara_capital = cidade.str.match(padrao_capital, na=False)
        rel["registros_capital"] = int(mascara_capital.sum())
    else:
        mascara_capital = pd.Series(True, index=df_cru.index)
        rel["registros_capital"] = len(df_cru)
        rel["aviso_cidade"] = "coluna de cidade ausente; recorte geografico nao aplicado"

    # --- rotulo --------------------------------------------------------------
    alvo = _rotular(df_cru[mapa["rubrica"]])
    rel["descartados_rubrica_ambigua"] = int(alvo.isna().sum())

    # --- datas e horas -------------------------------------------------------
    data_oc = _parse_data(df_cru[mapa["data_ocorrencia"]])
    df["dia_semana"] = data_oc.dt.dayofweek.map(
        {0: "SEGUNDA", 1: "TERCA", 2: "QUARTA", 3: "QUINTA",
         4: "SEXTA", 5: "SABADO", 6: "DOMINGO"}
    )
    df["fim_de_semana"] = (data_oc.dt.dayofweek >= 5).astype(float)
    df["mes"] = data_oc.dt.month.astype(float)

    if "hora_ocorrencia" in mapa:
        hora = _parse_hora(df_cru[mapa["hora_ocorrencia"]])
    else:
        hora = pd.Series(np.nan, index=df_cru.index)
    df["hora"] = hora
    # hora e circular: 23h e 0h sao vizinhas. Seno/cosseno preservam isso.
    df["hora_sen"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    if "periodo" in mapa:
        per = df_cru[mapa["periodo"]].map(_normaliza_valor).replace("", np.nan)
    else:
        per = pd.Series(np.nan, index=df_cru.index)
    # preenche o periodo a partir da hora quando o campo vier vazio
    per_da_hora = pd.cut(hora, bins=[-0.1, 5.9, 11.9, 17.9, 23.9],
                         labels=["MADRUGADA", "PELA MANHA", "A TARDE", "A NOITE"])
    df["periodo"] = per.fillna(pd.Series(per_da_hora, index=df_cru.index).astype(object))

    # --- atraso ate a comunicacao -------------------------------------------
    # Derivado de datas, nao da natureza do fato: nao ha vazamento do rotulo.
    if "data_comunicacao" in mapa:
        data_com = _parse_data(df_cru[mapa["data_comunicacao"]])
        atraso = (data_com - data_oc).dt.days
        df["dias_ate_comunicacao"] = atraso.where(atraso.between(0, 365))
    else:
        df["dias_ate_comunicacao"] = np.nan

    # --- categoricas de local e objeto --------------------------------------
    if "bairro" in mapa:
        b = df_cru[mapa["bairro"]].map(_normaliza_valor).replace("", np.nan)
        df["bairro"] = _agrupar_raros(b.fillna("NAO INFORMADO"), top_bairros)
    else:
        df["bairro"] = "NAO INFORMADO"

    if "tipo_local" in mapa:
        tl = df_cru[mapa["tipo_local"]].map(_normaliza_valor).replace("", np.nan)
        df["tipo_local"] = _agrupar_raros(tl.fillna("NAO INFORMADO"), 12)
    else:
        df["tipo_local"] = "NAO INFORMADO"

    if "marca_objeto" in mapa:
        mo = df_cru[mapa["marca_objeto"]].map(_normaliza_valor).replace("", np.nan)
        df["marca_objeto"] = _agrupar_raros(mo.fillna("NAO INFORMADA"), top_marcas)
    else:
        df["marca_objeto"] = "NAO INFORMADA"

    df["alvo"] = alvo

    # --- identificador do boletim (so para deduplicar, nunca como entrada) ---
    mapa_id = _mapear(df_cru, CANDIDATOS_ID)
    if "num_bo" in mapa_id:
        partes_id = [df_cru[mapa_id[k]].map(_normaliza_valor)
                     for k in ("ano_bo", "num_bo", "delegacia") if k in mapa_id]
        id_bo = partes_id[0]
        for p in partes_id[1:]:
            id_bo = id_bo + "|" + p
        # B.O. sem numero nao pode ser agrupado com nada: recebe id unico
        sem_numero = df_cru[mapa_id["num_bo"]].isna()
        id_bo = id_bo.where(~sem_numero, "SEM_BO_" + df_cru.index.astype(str))
        rel["chave_boletim"] = [mapa_id[k] for k in ("ano_bo", "num_bo", "delegacia")
                                if k in mapa_id]
    else:
        id_bo = None

    # --- aplica filtros ------------------------------------------------------
    mascara = mascara_capital & alvo.notna()
    df = df[mascara].copy()
    rel["apos_recorte_e_rotulo"] = len(df)

    # duplicidade: um mesmo B.O. pode listar varios celulares, gerando uma
    # linha por aparelho para o MESMO fato. A unidade de analise e a
    # ocorrencia, entao fica uma linha por boletim.
    #
    # Importante: NAO deduplicar pela igualdade dos atributos do modelo. Eles
    # sao grossos (bairro agrupado, hora cheia, marca agrupada) e duas
    # ocorrencias diferentes colidem com frequencia; descartar essas colisoes
    # apagaria justamente as combinacoes mais comuns, que carregam o sinal.
    antes = len(df)
    if id_bo is not None:
        df = df[~id_bo[mascara].duplicated(keep="first").to_numpy()]
        rel["criterio_duplicidade"] = "numero do boletim"
    else:
        rel["criterio_duplicidade"] = "nenhum (arquivo sem numero de B.O.)"
    rel["duplicatas_removidas"] = antes - len(df)

    # registros sem hora E sem periodo nao sustentam os atributos temporais
    df = df[~(df["hora"].isna() & df["periodo"].isna())]
    rel["apos_limpeza_temporal"] = len(df)

    if amostra_max is not None and len(df) > amostra_max:
        df = df.sample(n=amostra_max, random_state=semente)
        rel["amostragem"] = f"amostra aleatoria de {amostra_max} registros"

    df = df.reset_index(drop=True)
    rel["registros_finais"] = len(df)
    rel["proporcao_roubo"] = float(df["alvo"].mean())
    return df, rel


# --------------------------------------------------------------------------
# Entrada unica usada pelo notebook
# --------------------------------------------------------------------------

COLUNAS_CATEGORICAS = ["periodo", "dia_semana", "bairro", "tipo_local", "marca_objeto"]
COLUNAS_NUMERICAS = ["hora", "hora_sen", "hora_cos", "mes", "fim_de_semana", "dias_ate_comunicacao"]


def obter_dados(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Carrega a base oficial se existir; caso contrario usa a base espelho."""
    global ORIGEM_DOS_DADOS
    cru = carregar_base_ssp()
    if cru is not None and len(cru) > 0:
        ORIGEM_DOS_DADOS = "OFICIAL (SSP-SP, arquivos em data/raw/)"
    else:
        print("  nenhum arquivo em data/raw/ -> usando base espelho sintetica")
        cru = gerar_base_espelho()
        ORIGEM_DOS_DADOS = "ESPELHO SINTETICO (validacao de pipeline)"
    df, rel = preparar(cru, **kwargs)
    rel["origem"] = ORIGEM_DOS_DADOS
    return df, rel


if __name__ == "__main__":
    df, rel = obter_dados()
    print(rel)
    print(df.head())
    print(df["alvo"].value_counts())
