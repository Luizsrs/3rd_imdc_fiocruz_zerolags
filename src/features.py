import os
import pandas as pd
import numpy as np

def build_feature_matrix(target_file="dengue.csv.gz"):
    print("[*] Iniciando a construção da Matriz de Features...")
    
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    epi_path = os.path.join(raw_dir, target_file)
    df = pd.read_csv(epi_path)
    
    # 1. Transformação Municipal -> Estadual
    print("[*] Agregando dados de 5570 municípios para 27 Estados...")
    df['state_code'] = df['geocode'].astype(str).str[:2].astype(int)
    
    df_state = df.groupby(['state_code', 'date'])['casos'].sum().reset_index()
    
    df_state['date'] = pd.to_datetime(df_state['date'])
    df_state = df_state.sort_values(by=['state_code', 'date']).reset_index(drop=True)
    
    # 2. Remover Espírito Santo
    df_state = df_state[df_state['state_code'] != 32].copy()
    print(f"[+] Espírito Santo removido. Estados restantes para treino: {df_state['state_code'].nunique()}")

    # 3. Engenharia de Features Temporais (Lags)
    print("[*] Gerando defasagens temporais (Lags de 1 a 4 semanas)...")
    lags = [1, 2, 3, 4]
    for lag in lags:
        df_state[f'casos_lag_{lag}'] = df_state.groupby('state_code')['casos'].shift(lag)

    # 4. Limpeza e Exportação
    df_clean = df_state.dropna().copy()
    
    output_file = os.path.join(processed_dir, "feature_matrix_dengue_uf.parquet")
    df_clean.to_parquet(output_file, index=False)
    
    print(f"[+] Matriz ESTADUAL salva com sucesso em: {output_file}")
    print(f"[+] Total de semanas de histórico para treinamento: {len(df_clean)}")

if __name__ == "__main__":
    import os
    raw_dir = os.path.join("data", "raw")
    
    nome_exato = "dengue.csv.gz"
    caminho_completo = os.path.join(raw_dir, nome_exato)
    
    if os.path.exists(caminho_completo):
        print(f"[*] Base de Dados oficial: {nome_exato}")
        build_feature_matrix(nome_exato)
    else:
        print(f"[-] O arquivo '{nome_exato}' não foi encontrado.")
        print("[-] Rode o script data_download.py novamente.")