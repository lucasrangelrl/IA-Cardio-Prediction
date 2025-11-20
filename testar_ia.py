import joblib
import pandas as pd
import time

# 1. Carregar a IA
print(">>> Carregando o cérebro da IA...")
modelo = joblib.load('modelo_cardio_avancado.joblib')

# 2. Criando uma lista de pacientes diferentes para testar
fila_de_espera = [
    {
        "nome": "Paciente 1 (O Atleta)",
        "dados": {
            'age': 10000,       # ~27 anos
            'gender': 2,        # Homem
            'height': 175,
            'weight': 70.0,     # Peso ideal
            'ap_hi': 120,       # Pressão ótima
            'ap_lo': 80,
            'cholesterol': 1,   # Normal
            'gluc': 1,
            'smoke': 0,
            'alco': 0,
            'active': 1
        }
    },
    {
        "nome": "Paciente 2 (Risco Alto)",
        "dados": {
            'age': 21000,       # ~57 anos
            'gender': 1,        # Mulher
            'height': 160,
            'weight': 95.0,     # Obesidade
            'ap_hi': 160,       # Pressão Alta
            'ap_lo': 100,
            'cholesterol': 3,   # Muito Alto
            'gluc': 2,          # Açúcar alto
            'smoke': 0,
            'alco': 0,
            'active': 0
        }
    },
    {
        "nome": "Paciente 3 (Fumante Estressado)",
        "dados": {
            'age': 18000,       # ~49 anos
            'gender': 2,        # Homem
            'height': 180,
            'weight': 85.0,     # Leve sobrepeso
            'ap_hi': 140,       # Hipertensão leve
            'ap_lo': 90,
            'cholesterol': 2,   # Acima do normal
            'gluc': 1,
            'smoke': 1,         # Fuma
            'alco': 1,          # Bebe
            'active': 1
        }
    }
]

# 3. Função que processa cada paciente
def examinar_paciente(paciente_info):
    nome = paciente_info['nome']
    dados_brutos = paciente_info['dados']
    
    # Transforma dicionário em DataFrame
    df = pd.DataFrame([dados_brutos])

    # --- ENGENHARIA DE FEATURES (A mesma do treino) ---
    df['age_years'] = (df['age'] / 365.25).round().astype(int)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

    # Ordenar colunas (Gradient Boosting exige ordem exata)
    colunas = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 
        'age_years', 'bmi', 'pulse_pressure'
    ]
    df = df[colunas]

    # Previsão
    predicao = modelo.predict(df)[0]
    probabilidade = modelo.predict_proba(df)[0]
    
    risco_pct = probabilidade[1] * 100
    
    print(f"\n--- RELATÓRIO: {nome} ---")
    print(f"Dados: {dados_brutos['age']//365} anos, IMC: {df['bmi'].values[0]:.1f}, Pressão: {dados_brutos['ap_hi']}/{dados_brutos['ap_lo']}")
    
    if predicao == 1:
        print(f"🚨 DIAGNÓSTICO: ALTO RISCO ({risco_pct:.2f}% de chance)")
    else:
        print(f"✅ DIAGNÓSTICO: BAIXO RISCO (Apenas {risco_pct:.2f}% de chance de doença)")
    
    time.sleep(1) # Pausa para leitura

# 4. Loop principal
print(f"\nIniciando triagem de {len(fila_de_espera)} pacientes...")
for p in fila_de_espera:
    examinar_paciente(p)

print("\n=== FIM DA TRIAGEM ===")