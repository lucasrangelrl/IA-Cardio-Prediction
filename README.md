# 🫀 Cardio Prediction AI - Detecção de Risco Cardiovascular

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Success-green)
![Scikit-Learn](https://img.shields.io/badge/ML-GradientBoosting-orange)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

Este projeto é um sistema de Inteligência Artificial desenvolvido para prever a presença de doenças cardiovasculares em pacientes com base em dados clínicos e exames laboratoriais.

Diferente de scripts de análise simples, este projeto implementa uma arquitetura completa de **Engenharia de Machine Learning**, separando o treinamento do modelo (pipeline offline) da inferência em tempo real via **API REST**.

---

## 🎯 Objetivo
Criar um modelo preditivo capaz de classificar se um paciente possui alto risco cardíaco ou não, utilizando um dataset histórico de 70.000 pacientes reais.

O sistema não apenas classifica (Sim/Não), mas também fornece a **probabilidade** (certeza do modelo) e explica os fatores de risco (Feature Importance).

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Machine Learning:** Scikit-Learn (Gradient Boosting Classifier)
* **Manipulação de Dados:** Pandas, NumPy
* **API / Backend:** FastAPI, Uvicorn
* **Visualização:** Matplotlib, Seaborn
* **Persistência:** Joblib (Serialização do modelo)

---

## 📊 Engenharia de Dados e Modelo

Para superar a acurácia base de modelos simples, foi realizado um processo rigoroso de tratamento de dados:

### 1. Limpeza (Data Cleaning)
Dados clinicamente impossíveis foram removidos para evitar ruído no treinamento:
* Pressão sistólica/diastólica negativas ou fora da escala humana.
* Alturas e pesos inconsistentes (ex: adultos com 50cm de altura).

### 2. Feature Engineering (Criação de Variáveis)
Novas colunas foram calculadas matematicamente para aumentar a inteligência do modelo:
* **IMC (Índice de Massa Corporal):** Calculado a partir de peso e altura.
* **Pressão de Pulso:** A diferença entre a pressão sistólica e diastólica (indicador de rigidez arterial).
* **Idade em Anos:** Conversão da idade original (em dias).

### 3. O Algoritmo
Foi utilizado o **Gradient Boosting Classifier**. Este algoritmo constrói árvores de decisão sequenciais, onde cada nova árvore tenta corrigir os erros da anterior, resultando em uma precisão superior ao Random Forest tradicional.

---

## 📈 Resultados Obtidos

O modelo foi treinado com 80% dos dados e validado em 20% (dados nunca vistos).

| Métrica | Resultado | Descrição |
| :--- | :--- | :--- |
| **Acurácia Global** | **~73.80%** | Porcentagem de acertos totais. |
| **Recall (Doentes)** | Alto | Capacidade de detectar quem realmente está doente. |
| **Precision** | Equilibrada | Evita excesso de alarmes falsos. |

### Matriz de Confusão e Importância das Features
*(As imagens geradas pelo script `main.py` podem ser visualizadas na pasta raiz do projeto)*.

O modelo identificou que **Pressão Arterial**, **Idade** e **Colesterol** são os fatores mais determinantes para o diagnóstico.

---

## 📂 Estrutura do Projeto

```bash
📁 IA_Cardio
│
├── 📄 main.py              # Script de ETL e Treinamento do Modelo
├── 📄 api.py               # Servidor API (FastAPI) para uso em produção
├── 📄 testar_ia.py         # Script cliente que simula um sistema hospitalar
├── 📄 cardio_train.csv     # Dataset (Kaggle)
├── 🧠 modelo_cardio_avancado.joblib  # O "cérebro" da IA salvo
├── 📄 requirements.txt     # Dependências do projeto
└── 📄 README.md            # Documentação