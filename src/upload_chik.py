import os
import subprocess
import pandas as pd
from mosqlient import upload_prediction


API_KEY = "zero_lags:f46c7c08-f09c-4634-b87b-6e4c7282e7e6" 
REPOSITORY = "Luizsrs/3rd_imdc_fiocruz_zerolags" 

MAPA_IBGE = {
    'AC': 12, 'AL': 27, 'AP': 16, 'AM': 13, 'BA': 29, 'CE': 23, 'DF': 53, 'GO': 52, 
    'MA': 21, 'MT': 51, 'MS': 50, 'MG': 31, 'PA': 15, 'PB': 25, 'PR': 41, 'PE': 26, 
    'PI': 22, 'RJ': 33, 'RN': 24, 'RS': 43, 'RO': 11, 'RR': 14, 'SC': 42, 'SP': 35, 
    'SE': 28, 'TO': 17
}

def pegar_hash_do_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "dummy_hash"

def main():
    commit = pegar_hash_do_commit()
    print(f"[+] Iniciando transmissão de CHIKUNGUNYA para o Commit Git: {commit}")

    
    arquivos_validador = [
        "results/submissions/imdc_val1_chik_submission.csv",
        "results/submissions/imdc_val2_chik_submission.csv",
        "results/submissions/imdc_val3_chik_submission.csv",
        "results/submissions/imdc_val4_chik_submission.csv"
    ]

    for caminho_csv in arquivos_validador:
        if not os.path.exists(caminho_csv):
            print(f"[-] Arquivo não encontrado: {caminho_csv}. Garanta que rodou o chik_pipeline.py primeiro!")
            continue

        nome_base = os.path.basename(caminho_csv)
        print(f"\n[+] Enviando dados do arquivo opcional: {nome_base}")
        df = pd.read_csv(caminho_csv)

        for uf, df_uf in df.groupby('location'):
            if uf not in MAPA_IBGE:
                continue

            codigo_ibge = MAPA_IBGE[uf]
            colunas_exigidas = [
                'date', 'lower_95', 'lower_90', 'lower_80', 'lower_50', 
                'pred', 'upper_50', 'upper_80', 'upper_90', 'upper_95'
            ]
            
            dados_json = df_uf[colunas_exigidas].to_dict(orient='records')

            try:
                upload_prediction(
                    api_key=API_KEY,
                    disease="A92.0",
                    repository=REPOSITORY,
                    description=f"Validação Opcional Chikungunya - {nome_base}",
                    commit=commit,
                    case_definition="probable",
                    published=True,
                    adm_level=1,
                    adm_0="BRA",
                    adm_1=codigo_ibge,
                    prediction=dados_json
                )
                print(f"  [✓] Estado {uf} (Chikungunya) enviado!")
            except Exception as erro:
                if "Duplication found" in str(erro):
                    print(f"  [✓] Estado {uf} (Chikungunya) já constava no banco.")
                else:
                    print(f"  [-] Falha no estado {uf}: {erro}")

    print("\n[✓] Transmissão de Chikungunya finalizada!")

if __name__ == "__main__":
    main()