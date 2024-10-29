
from ultralytics import YOLO
import os
import pickle
import pandas as pd
import cv2
import face_recognition
from datetime import datetime
import winsound

class DrivingAttention:
    # Inicializamos el contador de tiempo consecutivo de "tired"
    contador_cansado = 0
    umbral_alerta = 2  # Umbral de 2 segundos para generar alerta
    intervalo_frame = 0.1  # Suponemos que las predicciones se generan cada 100 ms (0.1 segundos)

    def __init__(self, model_path, database_path, data_file, data_frame_file, data_status):
        self.model = YOLO(model_path)
        self.data_file = data_file
        self.database_path = database_path
        self.data_frame_file = data_frame_file
        self.data_status = data_status

        # Verificar si la carpeta para guardar las caras conocidas existe, si no, crearla
        if not os.path.exists(database_path):
            os.makedirs(database_path)

        # Cargar los datos de rostros conocidos (si existen)
        if os.path.exists(data_file):
            with open(data_file, "rb") as f:
                self.known_face_encodings, self.known_face_ids = pickle.load(f)  # Cargar IDs junto con encodings
        else:
            self.known_face_encodings = []
            self.known_face_ids = []

        # Cargar datos de personas desde el CSV, si existe
        if os.path.exists(data_frame_file):
            self.people_data = pd.read_csv(data_frame_file)
        else:
            # Si no existe, crear un DataFrame vacío
            self.people_data = pd.DataFrame(columns=["ID", "Name", "Age", "Gender"])

        if os.path.exists(data_status):
            self.df = pd.read_csv(data_status)
        else:
            # Si no existe, crear un DataFrame vacío
            self.df = pd.DataFrame(columns=["id","timestamp", "status", "probability"])

    def start_video_capture(self):
        self.video_capture = cv2.VideoCapture(0)

    def read_video(self):    
        ret, self.frame = self.video_capture.read()

        if not ret or self.frame is None:
            self.frame=None
            print('No se pudo capturar el frame')

    # Función para guardar nuevos rostros en la base de datos
    def save_new_face(self, age, gender):#face_encoding, name, age, gender, frame, known_face_encodings, known_face_ids, data_file, database_path, data_frame_file):
        # Generar un ID único
        self.person_id = len(self.known_face_encodings) + 1
        self.known_face_encodings.append(self.face_encoding)
        self.known_face_ids.append(self.person_id)  # Guardar el ID en lugar del nombre

        # Guardar en un archivo los rostros conocidos y sus IDs
        with open(self.data_file, "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_ids), f)

        # Guardar la imagen de la persona en la carpeta específica con formato ID-Name
        image_path = f"{self.database_path}/{self.person_id}.jpg"
        cv2.imwrite(image_path, self.frame)

        # Crear un nuevo DataFrame para la nueva persona
        new_data = pd.DataFrame({"ID": [self.person_id], "Name": [self.name], "Age": [age], "Gender": [gender]})

        # Concatenar el nuevo DataFrame con el existente
        self.people_data = pd.concat([self.people_data, new_data], ignore_index=True)

        # Guardar el DataFrame en un archivo CSV
        self.people_data.to_csv(self.data_frame_file, index=False)

    def faceid_init(self):#frame, known_face_encodings, known_face_ids, data_file, database_path, data_frame_file, people_data):
        # Convertir la imagen de BGR (OpenCV) a RGB (face_recognition usa RGB)
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # Buscar los rostros en la imagen y codificarlos
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), self.face_encoding in zip(face_locations, face_encodings):
            # Comprobar si el rostro es conocido
            matches = face_recognition.compare_faces(self.known_face_encodings, self.face_encoding)

            person_id = None
            name = "Desconocido"

            # Si hay coincidencias
            if True in matches:
                first_match_index = matches.index(True)
                self.person_id = self.known_face_ids[first_match_index]  # Obtener el ID correspondiente
                # Acceder al nombre usando el ID
                self.name = self.people_data.loc[self.people_data['ID'] == self.person_id, 'Name'].values[0]  # Obtener el nombre usando el ID
            else:
                # Si es un rostro nuevo, preguntar el nombre, edad y género
                self.name = input("No te reconozco, ¿Cuál es tu nombre?: ")
                age = input("¿Cuál es tu edad?: ")
                gender = input("¿Cuál es tu género?: (M o F)")

                # Guardar el nuevo rostro
                self.people_data = self.save_new_face(age, gender)#face_encoding, name, age, gender, frame, known_face_encodings, known_face_ids, data_file, database_path, data_frame_file)

    # Función para guardar las predicciones en el DataFrame
    def save_prediction(self, prediccion, probabilidad):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nueva_fila = pd.DataFrame([[self.person_id, timestamp, prediccion, probabilidad]], columns=["id", "timestamp", "status", "probability"])
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        return self.df

    # Función para lanzar una alerta sonora
    def alerta_sonora(self):
        print("¡ALERTA! El conductor ha estado cansado por más de 2 segundos consecutivos.")
        winsound.Beep(1000, 1000)  # Sonido a 1000 Hz durante 1 segundo

    def prediction(self):
        # Procesar el frame con YOLO
        results = self.model(self.frame)

        # Obtener las detecciones
        detections = results[0].names[results[0].probs.top1]
        probabilidad = results[0].probs.top1conf.item()  # Obtener la probabilidad más alta de la predicción

        # Guardar la predicción en el DataFrame
        self.df = self.save_prediction(detections, probabilidad)

        # Verificar si el estado actual es "tired"
        if detections == "tired":
            self.contador_cansado += self.intervalo_frame  # Aumentamos el contador en 0.1 segundos
        else:
            self.contador_cansado = 0  # Si no está cansado, reiniciamos el contador

        # Si el conductor ha estado "tired" por más de 2 segundos, generamos una alerta
        if self.contador_cansado >= self.umbral_alerta:
            self.alerta_sonora()
            self.contador_cansado = 0  # Reiniciar el contador tras la alerta para evitar repetidas alertas

        # Mostrar el estado en la ventana de video
        cv2.putText(self.frame, f'{self.name}: {detections}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def video_show(self):
        return cv2.imshow('Video', self.frame)
    
    def finish(self):
        self.video_capture.release()
        self.df.to_csv(self.data_status, index=False)

       
