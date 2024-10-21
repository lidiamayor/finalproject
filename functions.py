from datetime import datetime
import pickle
import pandas as pd
import cv2
import winsound
import face_recognition
 

# Función para guardar nuevos rostros en la base de datos
def save_new_face(face_encoding, name, age, gender, frame, known_face_encodings, known_face_ids, data_file, database_path, data_frame_file, people_data):
    # Generar un ID único
    person_id = len(known_face_encodings) + 1
    known_face_encodings.append(face_encoding)
    known_face_ids.append(person_id)  # Guardar el ID en lugar del nombre

    # Guardar en un archivo los rostros conocidos y sus IDs
    with open(data_file, "wb") as f:
        pickle.dump((known_face_encodings, known_face_ids), f)

    # Guardar la imagen de la persona en la carpeta específica con formato ID-Name
    image_path = f"{database_path}/{person_id}.jpg"
    cv2.imwrite(image_path, frame)

    # Crear un nuevo DataFrame para la nueva persona
    new_data = pd.DataFrame({"ID": [person_id], "Name": [name], "Age": [age], "Gender": [gender]})

    # Concatenar el nuevo DataFrame con el existente
    people_data = pd.concat([people_data, new_data], ignore_index=True)

    # Guardar el DataFrame en un archivo CSV
    people_data.to_csv(data_frame_file, index=False)

    return people_data


# Función para guardar las predicciones en el DataFrame
def save_prediction(prediccion, probabilidad, person_id, df):
    #global df
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nueva_fila = pd.DataFrame([[person_id, timestamp, prediccion, probabilidad]], columns=["id", "timestamp", "status", "probability"])
    df = pd.concat([df, nueva_fila], ignore_index=True)
    return df


# Función para lanzar una alerta sonora
def alerta_sonora():
    print("¡ALERTA! El conductor ha estado cansado por más de 2 segundos consecutivos.")
    winsound.Beep(1000, 1000)  # Sonido a 1000 Hz durante 1 segundo

def faceid_init(frame, known_face_encodings, known_face_ids, data_file, database_path, data_frame_file, people_data):
    # Convertir la imagen de BGR (OpenCV) a RGB (face_recognition usa RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Buscar los rostros en la imagen y codificarlos
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comprobar si el rostro es conocido
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        person_id = None
        name = "Desconocido"

        # Si hay coincidencias
        if True in matches:
            first_match_index = matches.index(True)
            person_id = known_face_ids[first_match_index]  # Obtener el ID correspondiente
            # Acceder al nombre usando el ID
            name = people_data.loc[people_data['ID'] == person_id, 'Name'].values[0]  # Obtener el nombre usando el ID
        else:
            # Si es un rostro nuevo, preguntar el nombre, edad y género
            name = input("No te reconozco, ¿Cuál es tu nombre?: ")
            age = input("¿Cuál es tu edad?: ")
            gender = input("¿Cuál es tu género?: (M o F)")

            # Guardar el nuevo rostro
            people_data = save_new_face(face_encoding, name, age, gender, frame, known_face_encodings, known_face_ids, data_file, database_path, data_frame_file, people_data)
    return name, person_id