from jproperties import Properties
import cvzone
from ultralytics import YOLO
import cv2
import os
import pandas as pd
import face_recognition
import numpy as np
from twilio.rest import Client
from datetime import datetime

configs = Properties()

with open("configuracao.properties", "rb") as config_file:
    configs.load(config_file)

arquivo_modelo = str(configs.get("arquivo_modelo").data)
know_faces_dir = str(configs.get("diretorio_foto_pessoas").data)
diretorio_modelo = str(configs.get("diretorio_modelo").data)
arquivo_modelo_full = os.path.join(diretorio_modelo, arquivo_modelo)
indice_camera = int(configs.get("indice_camera").data)
confianca = float(configs.get("confianca").data)
twilio_sid = str(configs.get("twilio_sid").data)
twilio_token = str(configs.get("twilio_token").data)
twilio_celular = str(configs.get("twilio_celular").data)
telefone_padrao = str(configs.get("telefone_padrao").data)
tempo_entre_envios_segundos = int(configs.get("tempo_entre_envios_segundos").data)

arquivo_base_pessoas = os.path.join(know_faces_dir, "pessoas.csv")

if not twilio_celular.startswith('+'):
    twilio_celular = '+' + twilio_celular
    
twilio_celular = str(twilio_celular).strip()    

df_pessoas = None

know_face_encodings = []
know_face_names = []

nome_janela = "TCC 2025 - Reconhecimento Facial" 

client = Client(twilio_sid, twilio_token)

#envio SMS
def enviar_sms(telefone_destino, mensagem):
    telefone_destino = str(telefone_destino).strip()
    
    print('from', twilio_celular)
    print('to', telefone_destino)
    
    message = client.messages.create(
        body=mensagem,
        from_=twilio_celular,
        to=telefone_destino    # Replace with the recipient's phone number
    )
    
    print(message.sid)
               
               
# Carregar imagens
def carregar_foto_pessoas():
    global know_face_encodings, know_face_names, df_pessoas
    
    if os.path.exists(arquivo_base_pessoas):
        df_pessoas = pd.read_csv(arquivo_base_pessoas, sep=';')
    else:
        df_pessoas = pd.DataFrame(columns=["arquivo", "nome", "telefone", "ultimo_envio_sms"])
        
    data_hora_inicio_2024 = datetime(2024, 1, 1, 10, 15, 16)
        
    for arquivo in os.listdir(know_faces_dir):
        if arquivo.endswith(".jpg") or arquivo.endswith(".png"):
            image_path = os.path.join(know_faces_dir, arquivo)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            
            know_face_encodings.append(encoding)
            know_face_names.append(arquivo)
            
            # Verifica se o arquivo já existe no DataFrame, se não, adiciona
            if df_pessoas[df_pessoas['arquivo'] == arquivo].empty:
                df_pessoas = pd.concat([df_pessoas, pd.DataFrame([{"arquivo": arquivo, 
                                                                    "nome": os.path.splitext(arquivo)[0], 
                                                                    "telefone": telefone_padrao, 
                                                                    "ultimo_envio_sms": data_hora_inicio_2024}])], ignore_index=True)
       
    df_pessoas.to_csv(arquivo_base_pessoas, sep=';', index=False)     
    
def get_telefone_pessoa(arquivo):
    global df_pessoas
    
    if df_pessoas is not None:
        pessoa = df_pessoas[df_pessoas['arquivo'] == arquivo]
        if not pessoa.empty:
            auxiliar = str(pessoa['telefone'].values[0])
            
            if not auxiliar.startswith('+'):
                auxiliar = '+' + auxiliar
            
            return auxiliar
    
    auxiliar = telefone_padrao
    
    if not auxiliar.startswith('+'):
        auxiliar = '+' + auxiliar
    
    return telefone_padrao  # Retorna o telefone padrão se não encontrar a pessoa

def get_nome_pessoa(arquivo):
    global df_pessoas
    
    if df_pessoas is not None:
        pessoa = df_pessoas[df_pessoas['arquivo'] == arquivo]
        if not pessoa.empty:
            return pessoa['nome'].values[0]
    
    return os.path.splitext(arquivo)[0]  # Retorna o nome do arquivo sem extensão se não encontrar a pessoa

def get_data_ultimo_envio_sms(arquivo):
    global df_pessoas
    
    if df_pessoas is not None:
        pessoa = df_pessoas[df_pessoas['arquivo'] == arquivo]
        if not pessoa.empty:
            return pessoa['ultimo_envio_sms'].values[0]
    
    return None

def set_ultimo_envio_sms(arquivo, data_envio):
    global df_pessoas
    
    if df_pessoas is not None:
        df_pessoas.loc[df_pessoas['arquivo'] == arquivo, 'ultimo_envio_sms'] = data_envio
        df_pessoas.to_csv(arquivo_base_pessoas, sep=';', index=False)
    else:
        print("DataFrame df_pessoas não está carregado.")

def existe_configuracao(arquivo):
    global df_pessoas
    
    if df_pessoas is not None:
        return not df_pessoas[df_pessoas['arquivo'] == arquivo].empty
    
    return False
            
def verificar_reconhecimento(frame, info):
    rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    retorno_faces_reconhecidas = []
    
    if face_locations is not None and face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encodings, face_encoding)
            name = "Desconhecido"
            
            face_distances = face_recognition.face_distance(know_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = know_face_names[best_match_index]
                retorno_faces_reconhecidas.append(name)
                
            face_names.append(name)
                
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            confidence = info.boxes.conf[0]
            
            if name == "Desconhecido":
                cor = (0, 0, 255)
                
                label = "Desconhecido"
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cor = (0, 255, 0) 
                
                nome_pessoa = get_nome_pessoa(name)
                label = f"{nome_pessoa} - {confidence*100:.2f}% Provavel"
                
                nome, prob = label.split(" - ")
                cv2.putText(frame, nome, (left + 6, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, prob, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            cv2.rectangle(frame, (left, top), (right, bottom), cor, 1)
            
    cv2.imshow(nome_janela, frame)
    
    return retorno_faces_reconhecidas
    
def main():
    # Executar Captura
    cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

    if not os.path.isfile(arquivo_modelo_full):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {arquivo_modelo_full}")

    # Carregar modelo YOLO para detecção de "objetos" faces - Detectar faces
    facemodel = YOLO(arquivo_modelo_full)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == 27:  # Tecla "Esc" para sair
            break
        else:
            frame = cv2.resize(frame, (700, 500))
            cv2.imshow(nome_janela, frame)

            face_result = facemodel.predict(frame, conf=confianca)
            
            if face_result is None or len(face_result) == 0 or face_result[0].boxes is None \
                or len(face_result[0].boxes) == 0:
                cvzone.putTextRect(frame, "Nenhum rosto detectado", (50, 50), scale=2, thickness=3, offset=10)
                cv2.imshow(nome_janela, frame)
            else:
                frame_original = frame.copy()
                
                for info in face_result:
                    # Reconhecimento facial
                    fotos_pessoas_reconhecidas = verificar_reconhecimento(frame_original, info)
                    
                    if fotos_pessoas_reconhecidas is None:
                        print('Nenhum rosto reconhecido.')
                    else:
                        for arquivo in fotos_pessoas_reconhecidas:
                            print(f"Rosto reconhecido: {arquivo}")
                            
                            if existe_configuracao(arquivo):
                                telefone_destino = get_telefone_pessoa(arquivo)
                                nome_pessoa = get_nome_pessoa(arquivo)
                                data_ultimo_envio_sms = get_data_ultimo_envio_sms(arquivo)
                                
                                if data_ultimo_envio_sms is None or (datetime.now() - pd.to_datetime(data_ultimo_envio_sms)).total_seconds() > tempo_entre_envios_segundos:
                                    try:
                                        enviar_sms(telefone_destino, f"Acesso de {nome_pessoa}, identificado nas dependências do ETEC Jaragua em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}.")
                                        set_ultimo_envio_sms(arquivo, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                    except Exception as e:
                                        print(f"Erro ao enviar SMS: {e}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    carregar_foto_pessoas()
    main()
