# api.py ATUALIZADO PARA V3
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cardio AI - V3 Advanced")

# Carregar modelo
try:
    modelo = joblib.load('modelo_cardio_avancado.joblib')
    print("Modelo V3 carregado!")
except:
    print("Erro ao carregar modelo. Rode o main.py primeiro.")

class PacienteRequest(BaseModel):
    age: int
    gender: int
    height: int
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

@app.post("/diagnosticar")
def diagnosticar(dados: PacienteRequest):
    # 1. Cria DataFrame com dados brutos
    input_dict = dados.dict()
    df = pd.DataFrame([input_dict])
    
    # 2. Aplica a Matemática (Feature Engineering)
    df['age_years'] = (df['age'] / 365.25).round().astype(int)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # 3. Ordena colunas
    colunas_necessarias = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 
        'age_years', 'bmi', 'pulse_pressure'
    ]
    df = df[colunas_necessarias]
    
    # 4. Previsão
    probabilidade = modelo.predict_proba(df)[0] # [prob_saudavel, prob_doente]
    risco_percentual = probabilidade[1] * 100
    
    tem_risco = risco_percentual > 50
    
    return {
        "diagnostico_texto": "RISCO CARDÍACO DETECTADO" if tem_risco else "Baixo Risco",
        "probabilidade_doenca": f"{risco_percentual:.2f}%",
        "dados_calculados": {
            "imc": round(df['bmi'].values[0], 2),
            "idade_anos": int(df['age_years'].values[0])
        }
    }