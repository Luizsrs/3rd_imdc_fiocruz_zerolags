import os
import pandas as pd
import numpy as np

def fix_quantile_crossing(input_file="raw_predictions_val1.parquet", output_name="imdc_val1_submission.csv"):
    print("[*] Iniciando a Varredura Anticorrupção...")
    
    processed_dir = os.path.join("data", "processed")
    data_path = os.path.join(processed_dir, input_file)
    
    if not os.path.exists(data_path):
        print("[-] Erro: Arquivo de predições brutas não encontrado.")
        return
        
    df = pd.read_parquet(data_path)
    
    q_cols = [col for col in df.columns if col.startswith('q_')]
    q_cols_sorted = sorted(q_cols, key=lambda x: float(x.split('_')[1]))
    
    print(f"[*] Verificando e corrigindo as colunas: {q_cols_sorted}")
    
    # 1. ORDENAÇÃO FORÇADA
    df[q_cols_sorted] = np.sort(df[q_cols_sorted].values, axis=1)
    
    # 2. ROMEAÇÃO DE COLUNAS PARA O PADRÃO OFICIAL DO MOSQLIMATE
    rename_map = {
        'q_0.025': 'lower_95',
        'q_0.05':  'lower_90',
        'q_0.1':   'lower_80',
        'q_0.25':  'lower_50',
        'q_0.5':   'pred',
        'q_0.75':  'upper_50',
        'q_0.9':   'upper_80',
        'q_0.95':  'upper_90',
        'q_0.975': 'upper_95'
    }
    
    df_submission = df.rename(columns=rename_map).copy()
    
    # ordem visual:
    final_columns = [
        'state_code', 'date', 'pred', 
        'lower_50', 'lower_80', 'lower_90', 'lower_95',
        'upper_50', 'upper_80', 'upper_90', 'upper_95'
    ]
    df_submission = df_submission[final_columns]
    
    # 3. VERIFICAÇÃO DE DOMINGOS
    non_sundays = (df_submission['date'].dt.dayofweek != 6).sum()
    if non_sundays > 0:
        print(f"[!] Encontradas {non_sundays} datas que não são domingos. Alinhar as datas.")
    else:
        print("[+] Todas as predições estão perfeitamente alinhadas nos domingos.")
        
    # Salva o arquivo em CSV
    df_submission['date'] = df_submission['date'].dt.strftime('%Y-%m-%d')
    output_path = os.path.join(processed_dir, output_name)
    df_submission.to_csv(output_path, index=False)
    
    print(f"\n[+] O arquivo de submissão está limpo e protegido.")
    print(f"[+] Salvo em: {output_path}")

if __name__ == "__main__":
    fix_quantile_crossing()