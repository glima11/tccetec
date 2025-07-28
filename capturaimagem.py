from jproperties import Properties
import cvzone
from ultralytics import YOLO
import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import pandas as pd
from datetime import datetime

configs = Properties()

with open("configuracao.properties", "rb") as config_file:
    configs.load(config_file)

arquivo_modelo = str(configs.get("arquivo_modelo").data)
diretorio_foto_pessoas = str(configs.get("diretorio_foto_pessoas").data)
diretorio_modelo = str(configs.get("diretorio_modelo").data)
arquivo_modelo_full = os.path.join(diretorio_modelo, arquivo_modelo)
indice_camera = int(configs.get("indice_camera").data)
telefone_padrao = str(configs.get("telefone_padrao").data)

arquivo_base_pessoas = os.path.join(diretorio_foto_pessoas, "pessoas.csv")

facemodel = YOLO(arquivo_modelo_full)

def gerar_csv():
    if os.path.exists(arquivo_base_pessoas):
        df_pessoas = pd.read_csv(arquivo_base_pessoas, sep=';')
    else:
        df_pessoas = pd.DataFrame(columns=["arquivo", "nome", "telefone", "ultimo_envio_sms"])
        
    data_hora_inicio_2024 = datetime(2024, 1, 1, 10, 15, 16)
        
    for arquivo in os.listdir(diretorio_foto_pessoas):
        if arquivo.endswith(".jpg") or arquivo.endswith(".png"):
            image_path = os.path.join(diretorio_foto_pessoas, arquivo)
            
            if df_pessoas[df_pessoas['arquivo'] == arquivo].empty:
                df_pessoas = pd.concat([df_pessoas, pd.DataFrame([{"arquivo": arquivo, 
                                                                    "nome": os.path.splitext(arquivo)[0], 
                                                                    "telefone": telefone_padrao, 
                                                                    "ultimo_envio_sms": data_hora_inicio_2024}])], ignore_index=True)
    
    df_pessoas.to_csv(arquivo_base_pessoas, sep=';', index=False)                 

def capturar_nome():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal

    # Abre uma caixa de diálogo solicitando o nome
    nome = simpledialog.askstring(title="Entrada de Nome", prompt="Digite seu nome:")

    if nome is not None:
        return nome
    else:
        print("Operação cancelada ou nenhuma entrada fornecida.")
        return None

def gravar_frame(frame):
    nome = capturar_nome()
    
    if nome is None or nome == "":
        print("Nenhum nome fornecido. Cancelando gravação.")
    else:
        filename = os.path.join(diretorio_foto_pessoas, f"{nome}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame gravado em: {filename}")

def main():
    cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

    facemodel = YOLO(arquivo_modelo_full)

    while True:
        ret, frame_original = cap.read()
        
        if not ret:
            break        
        
        key = cv2.waitKey(10)

        if key & 0xFF == 27:  # Tecla "Esc" para sair
            gerar_csv()
            break
        elif key & 0xFF == 13:  # Gravar
            novo_frame = frame_original.copy()
            
            novo_frame = cv2.resize(novo_frame, (700, 500))
            
            face_result = facemodel.predict(novo_frame, conf=0.40)
            
            if face_result is None or len(face_result) == 0 or face_result[0].boxes is None \
                or len(face_result[0].boxes) == 0:
                cvzone.putTextRect(novo_frame, 'Não detectado nenhuma face', (50, y_offset + i * 60), scale=2, thickness=3, offset=10)
                cv2.imshow("TCC 2025", novo_frame)
                cv2.namedWindow("TCC 2025", cv2.WINDOW_NORMAL)
                cv2.waitKey(1500)
            else:
                gravar_frame(novo_frame)
                print('gravar frame')
        else:
            novo_frame = cv2.resize(frame_original, (700, 500))

            lines = ["TCC 2025 ETEC - Cadastrar Acesso", "ESC para Sair", "ENTER para cadastrar"]
            y_offset = 50
            for i, line in enumerate(lines):
                cvzone.putTextRect(novo_frame, line, (50, y_offset + i * 60), scale=2, thickness=3, offset=10)
            cv2.imshow("TCC 2025", novo_frame)
            cv2.namedWindow("TCC 2025", cv2.WINDOW_NORMAL)
            #cv2.moveWindow("TCC 2025", 0, 0)

if __name__ == "__main__":
    main()