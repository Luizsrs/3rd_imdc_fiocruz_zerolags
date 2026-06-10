# Probabilistic Dengue Forecasting in Brazil - 3rd IMDC (2026)

Este repositório contém a infraestrutura completa de dados e modelagem preditiva submetida para a Fase de Validação do **3rd InfoDengue–Mosqlimate Dengue Challenge (IMDC 2026)**. 

O foco deste projeto é o Desafio Obrigatório (Dengue em Nível Estadual), empregando uma arquitetura de Regressão Quantílica baseada em árvores de decisão (*Gradient Boosting*), com tratamento matemático rigoroso para garantia de monotonicidade nos intervalos de confiança.

---

## 1. Team and Contributors
- **Líder Técnico / Cientista de Dados:** Luiz Carlos Soares da Costa
- **Instituição:** Fiocruz
- **Papel:** Arquitetura de dados, Feature Engineering, Modelagem Quantílica e Deploy do Pipeline.

## 2. Repository Structure
O projeto foi desenhado sob uma arquitetura de software modular, isolando scripts de processamento da base de dados local para garantir reprodutibilidade e prevenir *Data Leakage*.

* `data/` : Ignorado pelo versionamento do Git (configurado no .gitignore).
  * `raw/` : Arquivos brutos baixados via automação do FTP (ex: dengue.csv.gz).
  * `processed/` : Matrizes Parquet limpas e CSVs finais de submissão do desafio.
* `src/` : Pipeline de Machine Learning.
  * `data_download.py` : Automação de conexão e extração de dados do servidor FTP Mosqlimate.
  * `features.py` : Agregação municipal para nível UF e engenharia de defasagens temporais (lags).
  * `models.py` : Motor preditivo. Treinamento das 9 instâncias quantílicas do LightGBM por fold.
  * `post_process.py` : Filtro anticorrupção (Isotonic Sort) e formatação dos 4 folds exigidos.
* `.gitignore` : Bloqueio de artefatos de dados e cache de Python.
* `requirements.txt` : Pinagem de dependências de software.
* `README.md` : Documentação técnica do modelo.

## 3. Libraries and Dependencies
O ambiente preditivo foi construído em Python 3.12 utilizando um ecossistema isolado (Virtual Environment). Para reproduzir o modelo, instale as bibliotecas fixadas executando o comando abaixo no terminal:

pip install -r requirements.txt

A infraestrutura técnica (*Core Stack*) depende primariamente de: `pandas` (Manipulação de dados temporais), `numpy` (Cálculo vetorial e filtros matemáticos), `lightgbm` (Motor Preditivo) e `pyarrow` (Leitura e gravação otimizada de arquivos Parquet).

## 4. Data and Variables
A base de dados oficial (`dengue.csv.gz`), extraída diretamente da plataforma InfoDengue, foi submetida a um pipeline estrito de transformação e engenharia de variáveis:
1. **Agregação Geográfica:** Conversão estrutural via código de 5.570 municípios para 27 Unidades Federativas através do mapeamento dinâmico dos prefixos do IBGE (`geocode`).
2. **Exclusão de Escopo:** Remoção programática compulsória do Estado do Espírito Santo (código IBGE 32), conforme a regra técnica e mandatória do desafio.
3. **Memória Autoregressiva:** Criação das variáveis preditoras iterativas `casos_lag_1` a `casos_lag_4`, instruindo o algoritmo a capturar a inércia epidemiológica de curto prazo do ciclo de transmissão.

## 5. Model Training
O motor de inteligência artificial não busca a previsão da média pontual, mas sim a otimização direta da perda quantílica (*Pinball Loss*), essencial para a métrica de avaliação da competição.

Foi construído um modelo empilhado utilizando **LightGBM** com a função de custo configurada para `objective='quantile'`. O pipeline iterativo e automatizado processa os 4 cortes temporais retrospectivos da *Validation Round*, truncando os dados de treino na SE 25 de cada respectivo ano e prevendo a temporada real de dengue seguinte (da SE 41 à SE 40).

Para mapear a incerteza probabilística com precisão, a arquitetura treina independentemente **9 instâncias de regressão** para os percentis nominais exigidos pela banca: 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95 e 0.975.

## 6. Data Usage Restriction
Os conjuntos de dados e covariáveis mapeados neste repositório são provenientes de fontes abertas governamentais e do portal oficial de FTP do projeto Mosqlimate. Sua utilização neste projeto está estrita e exclusivamente reservada para fins de pesquisa, desenvolvimento científico e submissão na 3ª edição do *InfoDengue–Mosqlimate Dengue Challenge*.

## 7. Predictive Uncertainty
Como as 9 curvas quantílicas de probabilidade para cada Estado foram geradas por instâncias independentes de *Gradient Boosting*, existe o risco estatístico intrínseco de cruzamento de limites de incerteza nas caudas extremas da distribuição.

Para mitigar o erro crítico de **Quantile Crossing** e evitar a rejeição sumária pela métrica de avaliação WIS (*Weighted Interval Score*), desenvolveu-se o módulo dedicado de pós-processamento. Este script atua como uma barreira técnica aplicando:
1. **Isotonic Sort:** Uma reordenação monotônica forçada linha a linha (`np.sort(axis=1)`) sobre a matriz de quantis brutos preditos, garantindo de forma determinística a inequação fundamental do regulamento: lower_95 <= ... <= pred_50 <= ... <= upper_95.
2. **Non-negativity Constraint:** Aplicação de limites inferiores absolutos (`clip(lower=0)`) em toda a estrutura do DataFrame para impedir a propagação de previsões de casos abaixo de zero, ocasionadas por ruídos matemáticos do modelo em períodos de baixíssima transmissão.

## 8. References
- InfoDengue-Mosqlimate Dengue Challenge: Guia de Modelagem e Regras Oficiais (2026).
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.
- Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating epidemic forecasts in an interval format. *PLOS Computational Biology*, 17(2), e1008618.