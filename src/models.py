import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def train_all_validation_folds():
    print("[*] Iniciando a Arquitetura de Modelagem Quantílica para TODOS os Folds...")
    
    processed_dir = os.path.join("data", "processed")
    data_path = os.path.join(processed_dir, "feature_matrix_dengue_uf.parquet")
    
    if not os.path.exists(data_path):
        print("[-] Matriz de features não encontrada.")
        return
        
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    target = 'casos'
    features = [col for col in df.columns if 'lag' in col] 
    quantiles = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
    
    folds = [
        {'id': 1, 'train_end': '2022-06-25', 'test_start': '2022-10-09', 'test_end': '2023-10-08'},
        {'id': 2, 'train_end': '2023-06-25', 'test_start': '2023-10-08', 'test_end': '2024-10-06'},
        {'id': 3, 'train_end': '2024-06-25', 'test_start': '2024-10-06', 'test_end': '2025-10-05'},
        {'id': 4, 'train_end': '2025-06-25', 'test_start': '2025-10-05', 'test_end': '2026-10-04'}
    ]

    for fold in folds:
        print(f"\n=======================================================")
        print(f"[*] Processando Validação {fold['id']}...")
        
        train_df = df[df['date'] <= fold['train_end']].copy()
        test_df = df[(df['date'] >= fold['test_start']) & (df['date'] <= fold['test_end'])].copy()
        
        print(f"[*] Semanas de Treino: {len(train_df)} | Semanas para Prever: {len(test_df)}")
        
        if len(test_df) == 0:
            print(f" Sem dados de teste para o fold {fold['id']}. A base atual não vai até lá.")
            continue

        predictions = test_df[['state_code', 'date', target]].copy()
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        
        for q in quantiles:
            params = {
                'objective': 'quantile', 'alpha': q, 'metric': 'quantile',
                'learning_rate': 0.05, 'max_depth': 4, 'num_leaves': 15,
                'verbose': -1, 'seed': 42
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=100)
            
            pred_col_name = f'q_{q}'
            predictions[pred_col_name] = model.predict(X_test)
            predictions[pred_col_name] = predictions[pred_col_name].clip(lower=0)

        output_path = os.path.join(processed_dir, f"raw_predictions_val{fold['id']}.parquet")
        predictions.to_parquet(output_path, index=False)
        print(f"[+] Predições brutas da Validação {fold['id']} salvas!")

if __name__ == "__main__":
    train_all_validation_folds()