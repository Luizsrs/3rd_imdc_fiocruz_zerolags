import os
import pandas as pd
import numpy as np

def process_all_folds():
    print("[*] Iniciando a Varredura Anticorrupção para TODOS os Folds...\n")
    processed_dir = os.path.join("data", "processed")
    
    for fold_id in [1, 2, 3, 4]:
        input_file = f"raw_predictions_val{fold_id}.parquet"
        output_name = f"imdc_val{fold_id}_submission.csv"
        data_path = os.path.join(processed_dir, input_file)
        
        if not os.path.exists(data_path):
            continue
            
        print(f"[*] Limpando Quantile Crossing para: {input_file}...")
        df = pd.read_parquet(data_path)
        
        q_cols = [col for col in df.columns if col.startswith('q_')]
        q_cols_sorted = sorted(q_cols, key=lambda x: float(x.split('_')[1]))
        
        # 1. Ordenação Forçada
        df[q_cols_sorted] = np.sort(df[q_cols_sorted].values, axis=1)
        
        # 2. Renomeando
        rename_map = {
            'q_0.025': 'lower_95', 'q_0.05':  'lower_90', 'q_0.1':   'lower_80',
            'q_0.25':  'lower_50', 'q_0.5':   'pred',     'q_0.75':  'upper_50',
            'q_0.9':   'upper_80', 'q_0.95':  'upper_90', 'q_0.975': 'upper_95'
        }
        df_submission = df.rename(columns=rename_map).copy()
        
        final_columns = [
            'state_code', 'date', 'pred', 
            'lower_50', 'lower_80', 'lower_90', 'lower_95',
            'upper_50', 'upper_80', 'upper_90', 'upper_95'
        ]
        df_submission = df_submission[final_columns]
        
        df_submission['date'] = df_submission['date'].dt.strftime('%Y-%m-%d')
        output_path = os.path.join(processed_dir, output_name)
        df_submission.to_csv(output_path, index=False)
        
        print(f"[+] VALIDADO e Salvo: {output_name}\n")

if __name__ == "__main__":
    process_all_folds()