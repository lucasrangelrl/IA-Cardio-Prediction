import pandas as pd
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURAÇÕES ---
ARQUIVO_CSV = 'cardio_train.csv'
MODELO_ARQUIVO = 'modelo_cardio_avancado.joblib'

def feature_engineering_avancada(df):
    # 1. Converter idade de dias para anos (ajuda o modelo a achar padrões de faixa etária)
    df['age_years'] = (df['age'] / 365.25).round().astype(int)
    
    # 2. IMC (Body Mass Index)
    # Transforma altura em metros para o calculo
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # 3. Pressão de Pulso (Diferença entre Alta e Baixa)
    # Isso indica rigidez das artérias, forte indicador de risco
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    return df

def limpeza_medica_agressiva(df):
    print(f"Linhas antes da limpeza: {len(df)}")
    
    # Regra 1: Altura e Peso irreais
    # Remove anões/gigantes extremos (erros de digitação) e pesos impossíveis
    df = df[(df['height'] >= 140) & (df['height'] <= 210)]
    df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
    
    # Regra 2: Pressão Arterial Impossível
    # Sistólica não pode ser menor que Diastólica
    # Valores negativos ou absurdamente altos são erros de sensor/digitação
    df = df[df['ap_hi'] > df['ap_lo']] 
    df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 240)]
    df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 160)]
    
    # Regra 3: IMC Absurdo
    # Quem tem IMC < 10 ou > 60 ou é erro ou é outlier extremo que distorce a média
    df = df[(df['bmi'] > 15) & (df['bmi'] < 50)]
    
    print(f"Linhas após limpeza médica: {len(df)}")
    return df

def treinar_hardcore():
    if not os.path.exists(ARQUIVO_CSV):
        print("CSV não encontrado!")
        return

    print("1. Carregando dados...")
    df = pd.read_csv(ARQUIVO_CSV, sep=';')

    
    df = feature_engineering_avancada(df)
    df = limpeza_medica_agressiva(df)

   
    if 'id' in df.columns: df = df.drop('id', axis=1)
    
    X = df.drop('cardio', axis=1)
    y = df['cardio']

    # Separação
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Treinando Gradient Boosting (Isso é mais pesado que Random Forest)...")
    
    
    modelo = GradientBoostingClassifier(
        n_estimators=200,     
        learning_rate=0.1,    
        max_depth=5,          
        random_state=42
    )
    modelo.fit(X_train, y_train)

    # Avaliação
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"RESULTADO FINAL HARDCORE")
    print("="*40)
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print("-" * 40)

    
    joblib.dump(modelo, MODELO_ARQUIVO)
    print("Modelo salvo!")
    
    
    plt.figure(figsize=(10, 6))
    importancias = modelo.feature_importances_
    indices = pd.Series(importancias, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=indices, y=indices.index, hue=indices.index, legend=False, palette='magma')
    plt.title('O que realmente define o risco cardíaco?')
    plt.tight_layout()
    plt.savefig('grafico_v3_features.png')
    plt.close()

if __name__ == "__main__":
    treinar_hardcore()
