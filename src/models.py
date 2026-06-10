import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def train_quantile_models():
    print("[*] Iniciando a Arquitetura de Modelagem Quantílica (LightGBM)...")
    
    # 1. Carregar a Matriz de Features Limpa
    processed_dir = os.path.join("data", "processed")
    data_path = os.path.join(processed_dir, "feature_matrix_dengue_uf.parquet")
    
    if not os.path.exists(data_path):
        print("[-] Erro: Matriz de features não encontrada. Rode o features.py primeiro.")
        return
        
    df = pd.read_parquet(data_path)
    
    # 2. Definição Automática de Features e Alvo
    target = 'casos'
    features = [col for col in df.columns if 'lag' in col] 
    print(f"[+] Features utilizadas pelo modelo: {features}")
    
    # 3. Os 9 Quantis
    quantiles = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
    
    # 4. Split Temporal Estrito (Simulando o Teste de Validação 1)
    df['date'] = pd.to_datetime(df['date'])
    
    # Treino vai da SE 01/2010 até SE 25/2022
    train_df = df[df['date'] <= '2022-06-25'].copy()
    
    # Prever da SE 41/2022 até SE 40/2023 (Temporada de Dengue)
    test_df = df[(df['date'] >= '2022-10-09') & (df['date'] <= '2023-10-08')].copy()
    
    print(f"[*] Semanas de Treino: {len(train_df)} | Semanas para Prever (Teste): {len(test_df)}")
    
    # Preparamos o Dataframe
    predictions = test_df[['state_code', 'date', target]].copy()
    
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    
    # 5. O Motor Quantílico: Treinando 9 IAs separadas
    for q in quantiles:
        print(f"    -> Treinando motor para o quantil {q * 100}%...")
        
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'learning_rate': 0.05,
            'max_depth': 4,
            'num_leaves': 15,
            'verbose': -1,
            'seed': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Treinamento rápido e rasteiro
        model = lgb.train(params, train_data, num_boost_round=100)
        
        # Guarda a previsão na coluna específica
        pred_col_name = f'q_{q}'
        predictions[pred_col_name] = model.predict(X_test)
        
        # Filtro de Segurança
        predictions[pred_col_name] = predictions[pred_col_name].clip(lower=0)

    output_path = os.path.join(processed_dir, "raw_predictions_val1.parquet")
    predictions.to_parquet(output_path, index=False)
    print(f"\n 9 curvas de previsão salvas em: {output_path}")

if __name__ == "__main__":
    train_quantile_models()