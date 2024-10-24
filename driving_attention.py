from ultralytics import YOLO
import os
from pathlib import Path
import pickle
import pandas as pd
import cv2
from datetime import datetime
import streamlit as st
from PIL import Image, ImageEnhance
import random
import io
import numpy as np
#import threading
#import time
import face_recognition
import winsound
import shutil


class DrivingAttention:
    def __init__(self, database_path, data_file, data_frame_file, data_status, model_path, threshold):
        """Initialize the DrivingAttention system with improved error handling and configuration."""
        self.setup_paths(database_path, data_file, data_frame_file, data_status, model_path)
        self.threshold = threshold
        self.contador_cansado = 0
        self.umbral_alerta = 2
        self.intervalo_frame = 0.1
        self.current_status = None

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            raise Exception(f"Failed to load YOLO model: {str(e)}")
            
        self.load_face_data()
        self.load_people_data()

    def setup_paths(self, database_path, data_file, data_frame_file, data_status, model_path):
        """Setup and validate all required paths."""
        self.database_path = Path(database_path)
        self.data_file = Path(data_file)
        self.data_frame_file = Path(data_frame_file)
        self.data_status = Path(data_status)
        self.model_path = Path(model_path)
        
        # Create necessary directories
        self.database_path.mkdir(parents=True, exist_ok=True)

    def load_face_data(self):
        """Load face recognition data with error handling."""
        try:
            if self.data_file.exists():
                with open(self.data_file, "rb") as f:
                    self.known_face_encodings, self.known_face_ids = pickle.load(f)
            else:
                self.known_face_encodings = []
                self.known_face_ids = []
        except Exception as e:
            st.error(f"Error loading face data: {str(e)}")
            self.known_face_encodings = []
            self.known_face_ids = []

    def load_people_data(self):
        """Load people data with error handling."""
        try:
            if self.data_frame_file.exists():
                self.people_data = pd.read_csv(self.data_frame_file)
            else:
                self.people_data = pd.DataFrame(columns=["ID", "Name", "Age", "Gender", "Photo"])
                
            if self.data_status.exists():
                self.df = pd.read_csv(self.data_status)
            else:
                self.df = pd.DataFrame(columns=["id", "timestamp", "status", "probability"])
        except Exception as e:
            st.error(f"Error loading people data: {str(e)}")
            self.people_data = pd.DataFrame(columns=["ID", "Name", "Age", "Gender", "Photo"])
            self.df = pd.DataFrame(columns=["id", "timestamp", "status", "probability"])

    def start_video_capture(self):
        """Initialize video capture with error handling."""
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise Exception("Failed to open video capture")
        except Exception as e:
            raise Exception(f"Error initializing video capture: {str(e)}")

    def read_video(self):
        """Read video frame with error handling."""
        try:
            ret, self.frame = self.video_capture.read()
            if not ret or self.frame is None:
                self.frame = None
                raise Exception("Failed to capture frame")
        except Exception as e:
            st.error(f"Error reading video: {str(e)}")
            self.frame = None

    def faceid_init(self, placeholder):
        """Initialize face recognition and handle user registration."""
        # Convertir frame a RGB para face_recognition
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        take_photos = False
        # Detectar rostros
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            st.warning("No se detectó ningún rostro. Por favor, colócate frente a la cámara.")
            placeholder.image(self.video_show(), channels='RGB')
            return True, False

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        faceid=True
        for self.face_encoding in face_encodings:
            # Verificar si el rostro ya está registrado
            matches = face_recognition.compare_faces(self.known_face_encodings, self.face_encoding)
            
            if True in matches:
                # Usuario existente
                first_match_index = matches.index(True)
                self.person_id = self.known_face_ids[first_match_index]
                self.name = self.people_data.loc[self.people_data['ID'] == self.person_id, 'Name'].values[0]
                faceid = False
            else:
                # Nuevo usuario
                faceid = self.handle_new_user_registration()
        if faceid == False:
            taked_photos_path = 'data/taked_photos'
            new_photos_path = 'data/new_photos'
            categories = ['focus', 'tired', 'distracted']
            distribution = {'train': 0.65, 'val': 0.20, 'test': 0.15}
            take_photos = self.capture_images(taked_photos_path, new_photos_path, categories, distribution, placeholder)
        else:
            take_photos = True
                
        return faceid, take_photos

    def handle_new_user_registration(self):
        """Handle the registration process for new users."""

        col0, col1, col2 = st.columns([1, 3, 1])
    
        with col1:
            st.markdown("### 👤 Nuevo Usuario")
            st.write("Por favor, registra tus datos:")
            self.person_id = len(self.known_face_encodings) + 1

            with st.form("registro_usuario"):
                self.name = st.text_input("Nombre")
                age = st.text_input("Edad")
                gender = st.selectbox("Género", ["M", "F"])
                
                submitted = st.form_submit_button("Registrar")
                
                if submitted:
                    if not self.name or not age or not gender:
                        st.error("Por favor, completa todos los campos")
                        return True
                    
                    try:
                        age = int(age)
                        if age < 16 or age > 100:
                            st.error("La edad debe estar entre 16 y 100 años")
                            return True
                    except ValueError:
                        st.error("La edad debe ser un número")
                        return True
                    
                    # Guardar nuevo usuario
                    self.save_new_user(age, gender)
                    st.success(f"¡Bienvenido {self.name}! Usuario registrado correctamente.")
                    st.rerun()
                    return False
                
        return True

    def save_new_user(self, age, gender):
        """Save new user data."""
        # Crear nuevo registro
        new_data = pd.DataFrame({
            "ID": [self.person_id],
            "Name": [self.name],
            "Age": [age],
            "Gender": [gender],
            "Photo": [0]
        })
        
        # Actualizar DataFrame
        self.people_data = pd.concat([self.people_data, new_data], ignore_index=True)
        self.people_data.to_csv(self.data_frame_file, index=False)

        # Guardar la imagen de la persona en la carpeta específica con formato ID-Name
        image_path = f"{self.database_path}/{self.person_id}.jpg"
        cv2.imwrite(image_path, self.frame)

        # Guardar encoding facial si está disponible
        self.known_face_encodings.append(self.face_encoding)
        self.known_face_ids.append(self.person_id)
        
        with open(self.data_file, "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_ids), f)

    def save_prediction(self, detections, probabilidad):
        """Save prediction data."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nueva_fila = pd.DataFrame([[self.person_id, timestamp, detections, probabilidad]], 
                                columns=["id", "timestamp", "status", "probability"])
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        self.df.to_csv(self.data_status, index=False)

    def alerta_sonora(self):
        """Cross-platform alert system with cooldown."""
        self.alert_message = st.empty()
        self.alert_message.warning("¡ALERTA! El conductor muestra signos de cansancio.", icon="⚠️")
        winsound.Beep(1000, 1000) 
        self.alert_message = st.empty()


    def prediction(self):
        """Improved prediction with better error handling and status tracking."""
        try:
            results = self.model(self.frame)
            detections = results[0].names[results[0].probs.top1]
            probabilidad = results[0].probs.top1conf.item()
            
            if probabilidad > self.threshold:
                self.save_prediction(detections, probabilidad)
                self.current_status = detections
                
                if detections == "tired":
                    self.contador_cansado += self.intervalo_frame
                    if self.contador_cansado >= self.umbral_alerta:
                        self.alerta_sonora()
                        self.contador_cansado = 0
                else:
                    self.contador_cansado = 0
            
            # Add status overlay to frame
            self.add_status_overlay()
            
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            self.current_status = None

    def add_status_overlay(self):
        """Add modern status overlay to video frame."""
        if self.current_status:
            status_color = {
                'focus': (0, 255, 0),
                'tired': (0, 0, 255),
                'distracted': (255, 255, 0)
            }.get(self.current_status, (255, 255, 255))
            
            # Add semi-transparent overlay
            overlay = self.frame.copy()
            cv2.rectangle(overlay, (20, 20), (300, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
            
            # Add status text
            cv2.putText(self.frame, f'{self.name}: {self.current_status}',
                       (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    def get_current_status(self):
        """Get current driver status."""
        return self.current_status

    def video_show(self):
        """Convert frame to RGB for Streamlit display."""
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def finish(self):
        """Cleanup resources."""
        if hasattr(self, 'video_capture'):
            self.video_capture.release()
        self.df.to_csv(self.data_status, index=False)

    def capture_images(self, taked_photos_path, new_photos_path, categories, distribution, placeholder):
        """Capture and process images with improved UI and error handling."""
        try:
            number = self.people_data.loc[self.people_data['ID']==self.person_id, 'Photo'].values[0]
            
            if number >= 7:
                return True
            
            if number == 0:
                # Setup directories
                self.setup_photo_directories(taked_photos_path, new_photos_path, categories, distribution)
            
            # Determine current category
            category = self.get_current_category(categories, number)
            
            if number == 6:
                self.process_captured_images(taked_photos_path, new_photos_path, categories, distribution, placeholder)
                st.rerun()
                return True
                
            return self.handle_photo_capture(category, number, taked_photos_path)
            
        except Exception as e:
            st.error(f"Error en captura de imagen: {str(e)}")
            return False

    def setup_photo_directories(self, taked_photos_path, new_photos_path, categories, distribution):
        """Create necessary directories for photo storage."""
        if not hasattr(self, '_directories_created'):
            for category in categories:
                category_path = Path(f'{taked_photos_path}/{category}')
                # Verificar si el directorio existe
                if category_path.exists():
                    # Si el directorio existe, eliminar todo su contenido
                    shutil.rmtree(category_path)
                # Crear el directorio vacío
                category_path.mkdir(parents=True, exist_ok=True)
                
            for folder in distribution.keys():
                for category in categories:
                    category_path = Path(f'{new_photos_path}/{folder}/{category}')
                    # Verificar si el directorio existe
                    if category_path.exists():
                        # Si el directorio existe, eliminar todo su contenido
                        shutil.rmtree(category_path)
                    # Crear el directorio vacío
                    category_path.mkdir(parents=True, exist_ok=True)

            self._directories_created = True

    def get_current_category(self, categories, number):
        """Determine current photo category based on progress."""
        if number <= 1:
            return categories[0]
        elif number <= 3:
            return categories[1]
        else:
            return categories[2]

    def handle_photo_capture(self, category, number, taked_photos_path):
        """Handle the photo capture process with improved UI."""
        if 'photos' not in st.session_state:
            st.session_state.photos = []
        if 'current_photo' not in st.session_state:
            st.session_state.current_photo = number+1
        col0, col1, col2 = st.columns([1, 3, 1])
    
        with col1:
            if st.session_state.current_photo <= 6:
                photo = st.camera_input(f"Tomar foto {st.session_state.current_photo}/6 ({category})")
            
                if photo:
                    self.process_captured_photo(photo, category, taked_photos_path)
                    return False
            
        with col2:
            st.write("Captured photos:")
            cols = st.columns(2)
            for i, photo in enumerate(st.session_state.photos):
                print('indice: ',i)
                if (i+1)%2 != 0:
                    print('impar')
                    id = 0
                else:
                    print('par')
                    id = 1
                with cols[id]:
                    st.image(photo['thumbnail'], caption=f"Photo {i+1}", use_column_width=True)

            if st.session_state.current_photo > 6:
                st.success("All 6 photos have been captured and saved!")
           
        return False

    def process_captured_photo(self, photo, category, taked_photos_path):
        """Process and save captured photo."""
        try:
            img = Image.open(photo)
            resized_img = self.resize_image(img.copy())
            
            photo_path = f'{taked_photos_path}/{category}/{category}_{st.session_state.current_photo}.jpg'
            img.save(photo_path)
            
            buf = io.BytesIO()
            resized_img.save(buf, format='PNG')
            
            st.session_state.photos.append({
                'path': photo_path,
                'thumbnail': buf.getvalue()
            })
            
            self.mensaje_foto = st.empty()
            self.mensaje_foto.success(f"¡Foto {st.session_state.current_photo} guardada!")
            st.session_state.current_photo += 1
            self.people_data.loc[self.people_data['ID']==self.person_id, 'Photo'] += 1
            self.people_data.to_csv(self.data_frame_file, index=False)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error procesando foto: {str(e)}")

    def resize_image(self, image, max_size=(150, 150)):
        """Resize image maintaining aspect ratio."""
        image.thumbnail(max_size)
        return image

    def process_captured_images(self, taked_photos_path, new_photos_path, categories, distribution, placeholder):
        """Process all captured images and prepare for training."""
        try:
            self.augment_and_distribute_images(taked_photos_path, new_photos_path, categories, distribution, placeholder)
            self.people_data.loc[self.people_data['ID']==self.person_id, 'Photo'] += 1
            self.people_data.to_csv(self.data_frame_file, index=False)
            self.train_model()
            return True
        except Exception as e:
            st.error(f"Error procesando imágenes: {str(e)}")
            return False

    def augment_and_distribute_images(self, taked_photos_path, new_photos_path, categories, distribution, placeholder):
        """Augment and distribute images for training."""
        for category in categories:
            images = []
            for img_name in os.listdir(f'{taked_photos_path}/{category}'):
                img_path = f'{taked_photos_path}/{category}/{img_name}'
                img = cv2.imread(img_path)
                if img is not None:
                    images.append((img, img_name))

            for img, img_name in images:
                prefix = img_name.split(".")[0]
                self.augment_image(img, taked_photos_path, category, prefix)

            all_images = []
            for img_name in os.listdir(f'{taked_photos_path}/{category}'):
                img_path = f'{taked_photos_path}/{category}/{img_name}'
                img = cv2.imread(img_path)
                if img is not None:
                    all_images.append((img, img_name))

            random.shuffle(all_images)
            total_images = len(all_images)
            train_count = int(distribution['train'] * total_images)
            val_count = int(distribution['val'] * total_images)
            
            for img, img_name in all_images[:train_count]:
                cv2.imwrite(f'{new_photos_path}/train/{category}/{img_name}', img)
            for img, img_name in all_images[train_count:train_count + val_count]:
                cv2.imwrite(f'{new_photos_path}/val/{category}/{img_name}', img)
            for img, img_name in all_images[train_count + val_count:]:
                cv2.imwrite(f'{new_photos_path}/test/{category}/{img_name}', img)

    def augment_image(self, image, save_dir, category, prefix):
        """Apply image augmentation techniques."""
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Rotación
        for angle in [15, -15, 30, -30]:
            rotated = img.rotate(angle)
            rotated.save(f'{save_dir}/{category}/{prefix}_rotated_{angle}.jpg')

        # Inversión horizontal
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(f'{save_dir}/{category}/{prefix}_flipped.jpg')

        # Brillo
        enhancer = ImageEnhance.Brightness(img)
        for factor in [0.5, 1.5]:
            brightened = enhancer.enhance(factor)
            brightened.save(f'{save_dir}/{category}/{prefix}_brightness_{factor}.jpg')

        # Zoom
        width, height = img.size
        for crop_factor in [0.8, 0.6]:
            left = (1 - crop_factor) * width / 2
            top = (1 - crop_factor) * height / 2
            right = (1 + crop_factor) * width / 2
            bottom = (1 + crop_factor) * height / 2
            cropped = img.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
            cropped.save(f'{save_dir}/{category}/{prefix}_zoom_{crop_factor}.jpg')

    def train_model(self):
        """Train the model with new images."""
        self.mensaje_foto = st.empty()
        col0, col1, col2 = st.columns([1, 3, 1])
    
        with col1:
            with st.spinner(text="Entrenando el modelo..."):
                try:
                    model = YOLO(self.model_path)
                    model.train(data='data/new_photos', epochs=10, imgsz=128)
                    model.save(self.model_path)
                    st.success("¡Modelo entrenado exitosamente!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error entrenando modelo: {str(e)}")

