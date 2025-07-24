from jproperties import Properties
import cvzone
from ultralytics import YOLO
import cv2
import os
import face_recognition

configs = Properties()

with open("app-config.properties", "rb") as config_file:
    configs.load(config_file)

arquivo_modelo = str(configs.get("arquivo_modelo").data)
diretorio_origem = str(configs.get("diretorio_origem").data)
arquivo_modelo_full = os.path.join(diretorio_origem, arquivo_modelo)
indice_camera = int(configs.get("indice_camera").data)

cap = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

if not os.path.isfile(arquivo_modelo_full):
    raise FileNotFoundError(f"Arquivo do modelo não encontrado: {arquivo_modelo_full}")

facemodel = YOLO(arquivo_modelo_full)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if cv2.waitKey(10) & 0xFF == 27:  # Tecla "Esc" para sair
        break
    else:
        frame = cv2.resize(frame, (700, 500))
        cv2.imshow("TCC 2025", frame)

        face_result = facemodel.predict(frame, conf=0.40)
        
        if face_result is None or len(face_result) == 0 or face_result[0].boxes is None \
            or len(face_result[0].boxes) == 0:
            cvzone.putTextRect(frame, "Nenhum rosto detectado", (50, 50), scale=2, thickness=3, offset=10)
            cv2.imshow("TCC 2025", frame)
        else:
            for info in face_result:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = y2 - y1, x2 - x1
                    cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

                cvzone.putTextRect(
                    frame,
                    f"Faces: {len(face_result)}",
                    (50, 50),
                    scale=2,
                    thickness=3,
                    offset=10,
                )
                cv2.imshow("TCC 2025", frame)


cap.release()
cv2.destroyAllWindows()
