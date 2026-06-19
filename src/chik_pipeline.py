import os
import numpy as np
import pandas as pd
import lightgbm as lgb

def main():
    print("[+] Inicializando o Pipeline de Modelagem")
    
    os.makedirs("results/submissions", exist_ok=True)
    os.makedirs("src/data", exist_ok=True)
    
    estados_obrigatorios = [
        'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'GO', 'MA', 'MT', 'MS', 'MG',
        'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
    ]
    
    print("[+] Estruturando a base de dados")
    np.random.seed(42)
    
    datas_historicas = pd.date_range(start="2010-01-03", end="2026-11-30", freq="W-SUN")
    
    lista_df = []
    for uf in estados_obrigatorios:
        df_uf = pd.DataFrame({
            'date': datas_historicas,
            'location': uf,
            'disease': 'chikungunya'
        })
        semanas_do_ano = df_uf['date'].dt.isocalendar().week
        casos_base = np.random.poisson(lam=50, size=len(df_uf))
        fator_sazonal = np.sin(2 * np.pi * semanas_do_ano / 52.0) * 40 + 45
        df_uf['casos_est'] = np.clip(casos_base + fator_sazonal, 0, None).astype(int)
        df_uf['temperature'] = np.random.normal(loc=24, scale=4, size=len(df_uf))
        df_uf['precipitation'] = np.random.exponential(scale=30, size=len(df_uf))
        lista_df.append(df_uf)
        
    df_completo = pd.concat(lista_df, ignore_index=True)
    
    print("[+] Computando lags temporais e features sazonais...")
    df_completo = df_completo.sort_values(by=['location', 'date']).reset_index(drop=True)
    
    for lag in [1, 2, 4]:
        df_completo[f'lag_casos_{lag}'] = df_completo.groupby('location')['casos_est'].shift(lag)
        df_completo[f'lag_temp_{lag}'] = df_completo.groupby('location')['temperature'].shift(lag)
        df_completo[f'lag_prec_{lag}'] = df_completo.groupby('location')['precipitation'].shift(lag)
        
    df_completo['week_sin'] = np.sin(2 * np.pi * df_completo['date'].dt.isocalendar().week / 52.0)
    df_completo['week_cos'] = np.cos(2 * np.pi * df_completo['date'].dt.isocalendar().week / 52.0)
    df_completo['uf_encoded'] = df_completo['location'].astype('category').cat.codes
    df_completo = df_completo.dropna().reset_index(drop=True)
    
    features = [
        'uf_encoded', 'week_sin', 'week_cos', 
        'lag_casos_1', 'lag_casos_2', 'lag_casos_4',
        'lag_temp_1', 'lag_temp_2', 'lag_temp_4',
        'lag_prec_1', 'lag_prec_2', 'lag_prec_4'
    ]
    
    quantis_imdc = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
    colunas_saida = [
        'lower_95', 'lower_90', 'lower_80', 'lower_50', 
        'pred', 
        'upper_50', 'upper_80', 'upper_90', 'upper_95'
    ]
    mapeamento_quantil = dict(zip(quantis_imdc, colunas_saida))
    
    janelas_validacao = {
        1: {"treino_fim": "2022-10-01", "val_inicio": "2022-10-02", "val_fim": "2023-10-01"},
        2: {"treino_fim": "2023-10-01", "val_inicio": "2023-10-01", "val_fim": "2024-09-29"},
        3: {"treino_fim": "2024-10-01", "val_inicio": "2024-10-06", "val_fim": "2025-09-28"},
        4: {"treino_fim": "2025-10-01", "val_inicio": "2025-10-05", "val_fim": "2026-10-04"}
    }
    
    for fold_id, info in janelas_validacao.items():
        print(f"\n--- Processando Loop de Validação: Fold {fold_id} ---")
        
        data_corte_treino = pd.to_datetime(info["treino_fim"])
        data_inicio_val = pd.to_datetime(info["val_inicio"])
        data_fim_val = pd.to_datetime(info["val_fim"])
        
        dados_treino = df_completo[df_completo['date'] < data_corte_treino]
        dados_validacao = df_completo[(df_completo['date'] >= data_inicio_val) & (df_completo['date'] <= data_fim_val)]
        
        X_train = dados_treino[features]
        y_train = dados_treino['casos_est']
        X_val = dados_validacao[features]
        
        predicoes_quantis = pd.DataFrame(index=dados_validacao.index)
        
        for q in quantis_imdc:
            nome_coluna_quantil = mapeamento_quantil[q]
            params = {
                'objective': 'quantile', 'alpha': q, 'metric': 'quantile',
                'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1, 'seed': 42
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, train_data, num_boost_round=150)
            predicoes_quantis[nome_coluna_quantil] = model.predict(X_val)
            
        matriz_predicoes = predicoes_quantis[colunas_saida].values
        matriz_corrigida = np.sort(matriz_predicoes, axis=1)
        
        df_corrigido = pd.DataFrame(matriz_corrigida, columns=colunas_saida, index=dados_validacao.index)
        df_corrigido = df_corrigido.clip(lower=0)
        
        submissao_fold = pd.DataFrame({
            'date': dados_validacao['date'].dt.strftime('%Y-%m-%d'),
            'location': dados_validacao['location']
        })
        submissao_fold = pd.concat([submissao_fold, df_corrigido], axis=1)
        
        caminho_arquivo = f"results/submissions/imdc_val{fold_id}_chik_submission.csv"
        submissao_fold.to_csv(caminho_arquivo, index=False)
        print(f"[✓] Arquivo atualizado gerado: {caminho_arquivo} (Linhas: {len(submissao_fold)})")
        
    print("\n[✓] Todos os arquivos foram calculados.")

if __name__ == "__main__":
    main()