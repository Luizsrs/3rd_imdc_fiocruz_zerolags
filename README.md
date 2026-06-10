# Previsão Probabilística de Arboviroses - 3rd IMDC

## 1. Team and Contributors
- **Integrante:** Luiz Carlos Soares da Costa
- **Papel:** Cientista de Dados / Desenvolvedor Principal
- **Instituição:** Fiocruz

## 2. Repository Structure
Estrutura física do projeto projetada para garantir reprodutibilidade:
- `data/`: Armazenamento de dados locais (ignorado pelo versionamento).
- `src/`: Módulos em Python para automação do pipeline.
- `notebooks/`: Análises exploratórias e tunagem de hiperparâmetros.

## 3. Libraries and Dependencies
Ambiente construído em Python 3.10+ utilizando as bibliotecas especificadas no arquivo `requirements.txt`.

## 4. Data and Variables
Mapeamento de dados epidemiológicos (casos estimados via InfoDengue) cruzados com covariáveis climáticas locais (ERA5) e indicadores macroclimáticos globais (ENSO/IOD).

## 5. Model Training
Arquitetura quantílica avançada baseada no algoritmo LightGBM com otimização direta da perda por pinball (*Pinball Loss*) para estimativa de incerteza em 9 percentis.

## 6. Data Usage Restriction
Os dados utilizados neste projeto são de acesso público e destinados estritamente para fins de pesquisa epidemiológica e desenvolvimento das atividades do 3rd IMDC 2026.

## 7. Predictive Uncertainty
Mitigação do risco de *Quantile Crossing* através da aplicação de algoritmo de ordenação isotônica e imposição de restrição de não-negatividade nas caudas das curvas de probabilidade.

## 8. References
- Diretrizes oficiais e documentação do 3rd InfoDengue–Mosqlimate Dengue Challenge (IMDC 2026).
- Métrica de avaliação: Weighted Interval Score (WIS).