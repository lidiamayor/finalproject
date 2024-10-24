![DriveGuard Logo](https://github.com/lidiamayor/finalproject/blob/main/images/logo_driveguard_blue.png)

## 📋 Descripción del Proyecto

**DriveGuard** es una innovadora aplicación diseñada para prevenir accidentes de tráfico relacionados con la somnolencia al volante. Utilizando tecnologías avanzadas de visión por computadora, detección facial y monitoreo de fatiga, **DriveGuard** busca reducir drásticamente el riesgo de que los conductores se queden dormidos mientras conducen, salvando vidas en el proceso.

### ⚠️ Peligro de Dormirse al Volante
- Dormirse al volante es la **segunda causa de accidentes de tráfico** después del consumo de alcohol.
- **3 de cada 10 accidentes** son provocados por conductores que se quedan dormidos.
- La **somnolencia** estuvo presente en el **7% de los accidentes mortales de tráfico**.

## 💡 Solución Propuesta

### 1. 🧑‍💻 Detección Facial
Utilizando la librería **face_recognition**, extraemos características de la cara del conductor y las almacenamos en un archivo `.pkl`. A partir de estos datos, generamos un archivo **CSV** con la información de los conductores monitoreados.

### 2. 😴 Detección de Cansancio
Hemos construido una base de datos de imágenes etiquetadas para entrenar un modelo de clasificación de fatiga utilizando **YOLO (You Only Look Once)** de Ultralytics. El modelo fue entrenado inicialmente con 300 imágenes y posteriormente expandido a 750 imágenes, logrando una precisión del **88%** en la clasificación de estados de fatiga.

### 3. 🔄 Reentrenamiento del Modelo
Para mejorar la precisión y robustez del modelo, se generaron nuevas imágenes distorsionadas (rotaciones, cambios de brillo, etc.) utilizando la librería **PIL** (Python Imaging Library). Este reentrenamiento permitió un mejor ajuste del modelo para detectar la fatiga en diferentes condiciones de luz y ángulos.

### 4. 🚨 Alarma de Fatiga
Una vez que el sistema detecta que el conductor está cansado, se activa una alarma utilizando **winsound** para alertar al conductor y evitar un posible accidente.

### 5. 💻 Interfaz de Usuario
La interfaz del proyecto fue construida utilizando **Streamlit**, permitiendo la captura de fotos en tiempo real y mostrando los resultados de la detección de fatiga. También se incluye una integración con **OpenCV** para mostrar las imágenes del conductor en tiempo real.

## 🚀 Cómo Usar el Proyecto

1. **Instalación**:
   - Clona este repositorio:  
     `git clone https://github.com/usuario/driveguard.git`
   - Instala las dependencias requeridas:
     ```bash
     pip install -r requirements.txt
     ```

2. **Ejecución**:
   - Para iniciar la aplicación de detección de fatiga, ejecuta:
     ```bash
     streamlit run app.py
     ```
   - Sigue las instrucciones para tomar fotos y monitorear el estado del conductor.

3. **Visualización de Datos**:
   - Los datos generados se pueden analizar utilizando **Tableau**, donde se muestran los patrones de fatiga a lo largo del tiempo, distribuidos por hora del día y estación del año.

## 🎥 Videos Demostrativos

- **💻 Demostración de Streamlit**: [Ver vídeo](link-to-streamlit-video)
- **📊 Análisis de datos en Tableau**: [Ver vídeo](link-to-tableau-video)
- **🎤 Presentación del Proyecto**: [Ver presentación](link-to-presentation-video)

## 🛠️ Tecnologías Utilizadas

- **Python**: Desarrollo de la lógica principal.
- **Streamlit**: Interfaz de usuario para capturar fotos y mostrar los resultados.
- **YOLO (Ultralytics)**: Modelo de detección de fatiga.
- **Pillow (PIL)**: Manipulación de imágenes para el reentrenamiento.
- **face_recognition**: Detección de características faciales.
- **OpenCV**: Visualización en tiempo real de la cara del conductor.
- **Tableau**: Análisis visual de los datos recolectados.

## ✉️ Contacto

Si tienes alguna pregunta o sugerencia, no dudes en contactarme.

---

© 2024 DriveGuard
