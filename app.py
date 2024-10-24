import streamlit as st
import cv2
from pathlib import Path
from driving_attention import DrivingAttention
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="DriveGuard",
    page_icon="🚗",
    layout="wide"
)

# Crear directorios necesarios
BASE_DIR = Path("data")
INFO_DIR = BASE_DIR / "info-people"
for dir_path in [INFO_DIR, INFO_DIR / "images"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuración
CONFIG = {
    "database_path": str(INFO_DIR / "images"),
    "data_file": str(INFO_DIR / "face_data.pkl"),
    "data_frame_file": str(INFO_DIR / "people_data.csv"),
    "data_status": str(INFO_DIR / "status.csv"),
    "model_path": "model/model_3x2_.pt",
    "threshold": 0.55,
}

def main():
    # Estilos personalizados
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .stAlert {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Inicializar sistema
    try:
        da = DrivingAttention(**CONFIG)
        da.start_video_capture()
    except Exception as e:
        st.error(f"Error al inicializar el sistema: {str(e)}")
        return

    if not da.video_capture.isOpened():
        st.error("Error: No se encontró la cámara web. Por favor, verifica la conexión.")
        return

    # Diseño del dashboard
    col0, col1, col2 = st.columns([1, 3, 1])
    
    with col1:
        st.image('images/logo_driveguard_blue.png')
        placeholder = st.empty()  # Para el video
        
    with col2:
        st.subheader("Controles")
        start_button = st.toggle("Iniciar Monitoreo", key="start_monitoring")
        if start_button:
            st.info("Monitoreo activo")
        
        # Indicadores de estado
        titulo_container = st.empty()
        estado_container = st.empty()
        
    faceid = True
    take_photos = False

    if start_button:
        #try:
        da.read_video()
        #if da.frame is not None:
        faceid, take_photos = da.faceid_init(placeholder)
        
           
        while start_button and not faceid and take_photos:
            da.read_video()
            if da.frame is not None:
                da.prediction()
            
                # Actualizar estado usando los contenedores
                titulo_container.markdown(f"##### Estado del conductor/a {da.name}")
                status = da.get_current_status()
                if status:
                    status_color = {
                        'focus': '🟢 Atento',
                        'tired': '🔴 Cansado',
                        'distracted': '🟡 Distraído'
                    }.get(status, '⚪ Desconocido')
                    estado_container.markdown(f"{status_color}")

                
                # Mostrar video
                placeholder.image(da.video_show(), channels='RGB', use_column_width=True)
        #else:
        #    st.error("No hay imagen disponible de la cámara")
                
        #except Exception as e:
        #    st.error(f"Error durante el monitoreo: {str(e)}")'''
            
    # Limpieza
    da.finish()

if __name__ == "__main__":
    main()
