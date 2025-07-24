from jproperties import Properties
import cvzone
from ultralytics import YOLO
import cv2
import os
import face_recognition
import numpy as np

configs = Properties()

with open("app-config.properties", "rb") as config_file:
    configs.load(config_file)

arquivo_modelo = str(configs.get("arquivo_modelo").data)
know_faces_dir = str(configs.get("diretorio_foto_pessoas").data)
diretorio_modelo = str(configs.get("diretorio_modelo").data)
arquivo_modelo_full = os.path.join(diretorio_modelo, arquivo_modelo)
indice_camera = int(configs.get("indice_camera").data)
confianca = float(configs.get("confianca").data)

know_face_encodings = []
know_face_names = []

nome_janela = "TCC 2025 - Reconhecimento Facial" 

# Carregar imagens
def carregar_foto_pessoas():
    for filename in os.listdir(know_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(know_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            
            know_face_encodings.append(encoding)
            know_face_names.append(filename)
            
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
                label = "Desconhecido"
                cor = (0, 0, 255)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                label = f"{os.path.splitext(name)[0]} - {confidence*100:.2f}% Provavel"
                cor = (0, 255, 0) 
                
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
                        for nome in fotos_pessoas_reconhecidas:
                            print(f"Rosto reconhecido: {nome}")
                            # Colocar a chamada para por exemplo enviar notificação ou registrar acesso aqui

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    carregar_foto_pessoas()
    main()
