![DriveGuard Logo](https://github.com/lidiamayor/finalproject/blob/main/images/logo_driveguard_blue.png)

## 📋 Descripción del Proyecto

**DriveGuard** es una innovadora aplicación diseñada para prevenir accidentes de tráfico relacionados con la somnolencia al volante. Utilizando tecnologías avanzadas de visión por computadora, detección facial y monitoreo de fatiga, **DriveGuard** busca reducir drásticamente el riesgo de que los conductores se queden dormidos mientras conducen, salvando vidas en el proceso.

### ⚠️ Peligro de Dormirse al Volante
- Dormirse al volante es la **segunda causa de accidentes de tráfico** después del consumo de alcohol.
- **3 de cada 10 accidentes** son provocados por conductores que se quedan dormidos.
- La **somnolencia** estuvo presente en el **7% de los accidentes mortales de tráfico**.

## 💡 Solución Propuesta

### 1. 🧑‍💻 Creación de la Base de Datos
Para el monitoreo del estado del conductor, se creó una base de datos propia utilizando **Roboflow**, una plataforma que permite a los desarrolladores crear conjuntos de datos y modelos de visión computacional. Se tomaron aproximadamente 300 fotos de 5 personas diferentes (con distintos géneros y edades), las cuales fueron etiquetadas manualmente en tres categorías: **focus**, **distracted**, y **tired**. Posteriormente, se amplió la base de datos hasta alcanzar unas 740 imágenes utilizando técnicas de rotación, cambio de brillo y difuminado, lo que permite tener en cuenta las variaciones posibles en los vídeos, como la distancia del conductor a la cámara o las condiciones de iluminación.

### 2. 😴 Detección de Cansancio
Para lograr un modelo eficaz, se utilizó el algoritmo **YOLO (You Only Look Once)** de **Ultralytics**, yolov8s-cls, que es un modelo preentrenado en **ImageNet** para la clasificación de imágenes. Se optó por el modelo **small** para asegurar que no fuera muy pesado en el procesamiento de vídeo, logrando una precisión del **87%** en las predicciones (aproximadamente 10ms por predicción). Una vez entrenado el modelo, se guardó utilizando **Pytorch**, una librería de Machine Learning para aplicaciones de visión por computadora.

### 3. 🔄 Reentrenamiento del Modelo
El modelo se reentrena constantemente para mejorar su precisión. Para ello, se capturan nuevas imágenes del conductor usando **Streamlit**. Las imágenes son aumentadas rotándolas, invirtiéndolas, cambiando el brillo y haciendo zoom, pasando de tener 2 imágenes de cada tipo a 20. Finalmente, el modelo se ajusta nuevamente con las nuevas imágenes utilizando **YOLO**.

### 4. 🚨 Alarma de Fatiga
Cuando el sistema detecta que el conductor está cansado durante más de 2 segundos, se activa una alarma sonora utilizando **winsound**, que permite acceder a los mecanismos básicos de reproducción de sonido en Windows.

### 5. 💻 Interfaz de Usuario y Visualización
La interfaz del proyecto fue construida utilizando **Streamlit**, permitiendo la captura de fotos en tiempo real y mostrando los resultados de la detección de fatiga. Para visualizar y monitorear en tiempo real, se utilizó **OpenCV**, que es una librería optimizada para visión artificial. Además, los datos generados se recopilan y almacenan utilizando **Pandas** en un archivo .csv, y se visualizan en un dashboard interactivo en **Tableau**, facilitando la comprensión y análisis de los patrones de fatiga a lo largo del tiempo.


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

## 🎥 Demostrativos

- **💻 Demostración de Streamlit**: [Ver vídeo](https://www.canva.com/design/DAGUfQ6YOz4/dzUl5iwBRL0HAGdqNJ_8Zw/watch?utm_content=DAGUfQ6YOz4&utm_campaign=designshare&utm_medium=link&utm_source=editoro)
- **📊 Análisis de datos en Tableau**: [Ver dashboard](https://public.tableau.com/views/DriveGuard/Historia?:language=es-ES&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- **🎤 Presentación del Proyecto**: [Ver presentación](https://www.canva.com/design/DAGUeHJxpcc/KH67qUOGjCwiN0sOYC0sfA/view?utm_content=DAGUeHJxpcc&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## 🛠️ Tecnologías Utilizadas

- **Python**: Desarrollo de la lógica principal.
- **Roboflow**: Creación y ampliación de la base de datos de imágenes para el modelo de visión por computadora.
- **YOLO (Ultralytics)**: Modelo reentrenado para el monitoreo del conductor.
- **PyTorch**: Librería de Machine Learning utilizada para almacenar el modelo final.
- **OpenCV**: Visualización en tiempo real de la cara del conductor.
- **Winsound**: Activación de la alarma sonora cuando el conductor está cansado.
- **face_recognition**: Detección de características faciales.
- **Pickle**: Almacenamiento de codificaciones faciales y jerarquía de objetos de Python.
- **Streamlit**: Interfaz de usuario para capturar fotos y mostrar los resultados.
- **Pillow (PIL)**: Manipulación de imágenes para el reentrenamiento.
- **Pandas**: Recopilación y almacenamiento de datos en archivos .csv para su posterior análisis.
- **Tableau**: Análisis visual de los datos recolectados.

## ✉️ Contacto

Si tienes alguna pregunta o sugerencia, no dudes en contactarme.

---

© 2024 DriveGuard
