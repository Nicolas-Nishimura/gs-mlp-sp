/* Inferencia da MLP em JavaScript puro.
 *
 * Reproduz, na ordem exata, o que o ColumnTransformer e o modelo Keras fazem:
 *   1. one-hot das categoricas (na ordem de COLUNAS_CATEGORICAS,
 *      categorias em ordem alfabetica, como o OneHotEncoder do scikit-learn);
 *   2. padronizacao das numericas com a media e o desvio do StandardScaler;
 *   3. passagem pelas camadas Dense / BatchNormalization salvas no JSON.
 * Dropout nao aparece: e desligado na inferencia.
 */

function montarVetor(pacote, entrada) {
  const v = [];

  pacote.colunas_categoricas.forEach((col, i) => {
    const cats = pacote.categorias[i];
    let valor = entrada[col];
    if (valor === undefined || valor === null || valor === "") {
      valor = pacote.imput_cat[i];              // SimpleImputer(most_frequent)
    }
    valor = String(valor);
    // handle_unknown="ignore": categoria desconhecida vira um bloco de zeros
    cats.forEach((c) => v.push(c === valor ? 1 : 0));
  });

  pacote.colunas_numericas.forEach((col, i) => {
    let x = entrada[col];
    if (x === undefined || x === null || Number.isNaN(Number(x))) {
      x = pacote.imput_num[i];                  // SimpleImputer(median)
    }
    v.push((Number(x) - pacote.media[i]) / pacote.escala[i]);
  });

  return v;
}

function propagar(pacote, vetor) {
  let x = vetor;
  for (const camada of pacote.pesos) {
    if (camada.tipo === "dense") {
      const W = camada.W, b = camada.b;
      const saida = new Array(b.length).fill(0);
      for (let j = 0; j < b.length; j++) {
        let s = b[j];
        for (let i = 0; i < x.length; i++) s += x[i] * W[i][j];
        saida[j] = camada.ativacao === "relu" ? Math.max(0, s)
                 : camada.ativacao === "sigmoid" ? 1 / (1 + Math.exp(-s))
                 : s;
      }
      x = saida;
    } else if (camada.tipo === "batchnorm") {
      x = x.map((xi, i) =>
        camada.gamma[i] * (xi - camada.media[i]) /
        Math.sqrt(camada.var[i] + camada.eps) + camada.beta[i]
      );
    }
  }
  return x;
}

/** Deriva os atributos calculados e devolve P(roubo) e P(furto). */
function classificar(pacote, form) {
  const hora = Number(form.hora);
  const fimDeSemana = (form.dia_semana === "SABADO" || form.dia_semana === "DOMINGO") ? 1 : 0;

  const entrada = {
    periodo: form.periodo,
    dia_semana: form.dia_semana,
    bairro: form.bairro,
    tipo_local: form.tipo_local,
    marca_objeto: form.marca_objeto,
    hora: hora,
    hora_sen: Math.sin((2 * Math.PI * hora) / 24),
    hora_cos: Math.cos((2 * Math.PI * hora) / 24),
    mes: Number(form.mes),
    fim_de_semana: fimDeSemana,
    dias_ate_comunicacao: Number(form.dias_ate_comunicacao),
  };

  const pRoubo = propagar(pacote, montarVetor(pacote, entrada))[0];
  // limiar escolhido na validacao (Youden); 0,5 so como reserva
  const limiar = (pacote.metadados && pacote.metadados.limiar != null)
    ? pacote.metadados.limiar : 0.5;
  return { roubo: pRoubo, furto: 1 - pRoubo, limiar: limiar,
           classe: pRoubo >= limiar ? "ROUBO" : "FURTO" };
}

/** Periodo coerente com a hora informada, usado para preencher o formulario. */
function periodoDaHora(h) {
  if (h < 6) return "MADRUGADA";
  if (h < 12) return "PELA MANHA";
  if (h < 18) return "A TARDE";
  return "A NOITE";
}

if (typeof module !== "undefined") {
  module.exports = { montarVetor, propagar, classificar, periodoDaHora };
}