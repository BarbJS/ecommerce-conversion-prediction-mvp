# 🛒 QuickBite: E-commerce Conversion Prediction (MVP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lagpkR-jysShZHQmXirVjRjADdOcEwNp?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-172424?style=flat&logo=xgboost&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-MVP_Concluído-success.svg)

> **Motor preditivo de Machine Learning focado em identificar a propensão de compra de usuários em tempo real para otimização de campanhas de Marketing e ROI.**
> 
> 🎓 *Projeto de MVP (Minimum Viable Product) desenvolvido como requisito de conclusão da sprint de Machine Learning para a Pós-Graduação em Ciência de Dados & Analytics da **PUC-Rio**.*

---

## 🎯 O Desafio de Negócio

No ecossistema do **QuickBite**, aproximadamente **85% dos visitantes apenas navegam (vitrinistas) e apenas 15% efetivamente finalizam uma compra**. O desafio central deste MVP foi desenvolver um algoritmo capaz de identificar os 15% de compradores em potencial ainda durante a sessão ativa, permitindo que a equipe de Marketing dispare gatilhos proativos (como cupons de desconto limitados) de forma cirúrgica, sem desperdiçar a margem de lucro com clientes que não iriam converter.

---

## 🗂️ Sobre o Dataset (Origem e Contexto)

Os dados utilizados neste projeto são baseados no renomado **Online Shoppers Purchasing Intention Dataset**, disponibilizado pelo repositório oficial da UCI.

* **O que contém:** O dataset é composto por 12.330 sessões de navegação abstraídas. Ele mapeia a jornada do cliente combinando métricas do *Google Analytics* (como *BounceRates*, *ExitRates* e *PageValues*), volume de interação por categoria de página (Administrativa, Informativa ou Produtos) e informações contextuais de sessão (tipo de visitante, proximidade a datas promocionais e sistema operacional).
* **A Natureza do Desbalanceamento:** Refletindo a realidade severa do varejo online, o alvo do modelo (`Revenue`) é altamente desbalanceado: 10.422 sessões (84,5%) resultaram em abandono, enquanto apenas 1.908 (15,5%) terminaram em compra.
* **Autores:** C. Okan Sakar e Yomi Keles.
* **Acesso Oficial:** [UCI Machine Learning Repository - Online Shoppers Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
* **Licença de uso:** This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

---

## 🧠 A Solução Técnica e Arquitetura

O projeto supera o **Paradoxo da Acurácia**, cenário em que um modelo atinge 85% de acerto simplesmente classificando todos os acessos como "Abandono", tornando-se inútil para o negócio. Para garantir relevância preditiva, o modelo foi otimizado tendo o **F1-Score da classe minoritária (Conversão)** como norteador matemático.

A arquitetura final é construída sobre o **XGBoost (Extreme Gradient Boosting)**, aliando altíssimo poder de separabilidade não-linear com eficiência computacional para inferências de baixa latência (API Real-time).

### 🏆 Destaques do Projeto

* **Engenharia de Dados (EDA):** Comprovação empírica do poder de atributos técnicos de navegação (como `PageValues` e `BounceRates`) frente ao comportamento de consumo.
* **Mitigação de Desbalanceamento:** Utilização da tríade de regularização (Bagging nativo via `subsample`, penalização L2 Ridge e `class_weight='balanced'`) em detrimento de técnicas sintéticas custosas como SMOTE.
* **Otimização Inteligente:** *Fine Tuning* via `RandomizedSearchCV` acoplado a validação `StratifiedKFold`, respeitando restrições de *time-to-market* e custo computacional em nuvem.
* **Governança e MLOps:** Exportação persistente do modelo encapsulado no formato `.pkl`, preservando a integridade dos transformadores (*One-Hot Encoding* e *Standard Scaler*).

---

## 📊 Resultados e Performance

A evolução da modelagem demonstra claramente a superioridade da arquitetura de *Boosting* para lidar com dados altamente desbalanceados e relações não-lineares.

| Modelo | F1-Score (Conversão) | ROC-AUC | Precisão | Recall | Tempo de Treino | Tempo de Inferência |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Dummy) | 0.0000 | 0.5000 | 0.00 | 0.00 | 0.05s | 0.01s |
| Regressão Logística | 0.6083 | 0.8845 | 0.51 | 0.74 | 0.11s | ~0.01s |
| Random Forest | 0.5866 | 0.8980 | 0.72 | 0.49 | 1.91s | ~0.02s |
| Gradient Boosting | 0.6532 | 0.9304 | 0.72 | 0.59 | 4.02s | ~0.03s |
| XGBoost (Padrão) | 0.6268 | 0.9240 | 0.68 | 0.57 | 0.38s | ~0.03s |
| **XGBoost (Otimizado)** | **0.6523** | **0.9291** | **0.72** | **0.59** | **~3 min** *(Tuning)* | **0.03s** |

**Por que o XGBoost Otimizado foi escolhido como a solução final?**
O **XGBoost Otimizado** foi eleito o campeão absoluto por entregar o melhor equilíbrio entre viabilidade técnica e impacto de negócios. A *Regressão Logística* gerou excesso de falsos alarmes (baixa Precisão), enquanto a *Random Forest* se mostrou demasiadamente conservadora (baixo Recall), deixando compradores escaparem. Embora o *Gradient Boosting* clássico tenha atingido métricas estritamente similares, ele exigiu um tempo de processamento muito superior. O XGBoost, após o refinamento de hiperparâmetros, cravou um F1-Score maduro (0.6523) e um ROC-AUC de excelência (0.92+), garantindo alta taxa de acerto e tempos de inferência na casa dos milissegundos (0.03s), um requisito inegociável para um motor de propensão operando ao vivo no e-commerce.

---

## 🛠️ Como Executar o Projeto

Para reproduzir a pesquisa e a modelagem na sua máquina local:

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/BarbJS/ecommerce-conversion-prediction-mvp.git
   cd ecommerce-conversion-prediction-mvp

2. **Instale as dependências necessárias:**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

3. **Execute o Jupyter Notebook:**
   
Abra o arquivo quickbite_MVP.ipynb em seu ambiente de preferência (Jupyter, VSCode ou Google Colab) e execute as células sequencialmente. O modelo final e as métricas serão salvos automaticamente na sua máquina.


## 🚀 Próximos Passos (Evolução do MVP)

- Integração Explicável (SHAP): Adicionar painel de explicabilidade para justificar individualmente a propensão ao time de negócios.

- Threshold Tuning: Implementar painel interativo para a equipe de Marketing controlar a "agressividade" das ofertas de cupom (ajuste do limiar de probabilidade).

- Deploy e Teste A/B: Empacotar a predição via FastAPI ou Flask para testes isolados em 20% do tráfego orgânico do site.

**Autora:** Bárbara Jaeger Specian | **Ano:** 2026
