import os
import sys
from ftplib import FTP

def download_imdc_datasets():
    ftp_server = "info.dengue.mat.br"
    remote_directory = "data_imdc_2026"
    local_target_dir = os.path.join("data", "raw")
    
    os.makedirs(local_target_dir, exist_ok=True)
    
    print(f"[*] Iniciando conexão com o servidor FTP: {ftp_server}")
    
    try:
        ftp = FTP(ftp_server)
        ftp.login()
        print(f"Login efetuado com sucesso. Navegando para '{remote_directory}'...")
        ftp.cwd(remote_directory)
        
        
        all_remote_files = ftp.nlst()
        target_files = [f for f in all_remote_files if f.endswith(".gz") or f.endswith(".csv")]
        
        if not target_files:
            print("[-] Nenhum arquivo válido (.csv ou .gz) foi encontrado no diretório remoto.")
            ftp.quit()
            return
            
        print(f"[+] {len(target_files)} arquivos identificados para download.")
        
        for file_name in target_files:
            local_file_path = os.path.join(local_target_dir, file_name)
            
            if os.path.exists(local_file_path):
                print(f"[~] Arquivo '{file_name}' já existe localmente. Pulando...")
                continue
                
            print(f"[*] Baixando: {file_name} -> {local_file_path}...")
            
            with open(local_file_path, "wb") as local_file:
                # Transfere o arquivo em blocos binários de 8KB
                ftp.retrbinary(f"RETR {file_name}", local_file.write, blocksize=8192)
                
        ftp.quit()
        print("[+] Pipeline de ingestão concluído. Todos os dados brutos estão na pasta data/raw/.")
        
    except Exception as error:
        print(f"\n[-] Falha crítica na execução do pipeline de dados: {str(error)}")
        print("[-] Verifique sua conexão com a internet ou a estabilidade do servidor do InfoDengue.")
        sys.exit(1)

if __name__ == "__main__":
    download_imdc_datasets()