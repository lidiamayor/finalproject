import streamlit as st
import cv2
from driving_attention import DrivingAttention

# Ruta donde almacenaremos las imágenes y datos de las personas conocidas
database_path = "info-people/images"
data_file = "info-people/face_data.pkl"
# Archivo donde guardaremos los datos de las personas
data_frame_file = "info-people/people_data.csv"
data_status = "info-people/status.csv"
# Ruta del modelo
model_path = "../model/model_3x2.pt"

threshold = 0.55

da = DrivingAttention(model_path, database_path, data_file, data_frame_file, data_status, threshold)

def main():

    # Inicializar la cámara
    #video_capture = 
    da.start_video_capture()

    if not da.video_capture.isOpened():
        st.error("Error: Webcam not found.")
        return

    # Create the stop button
    start_button = st.toggle("Start")

    # Create an empty placeholder to show the webcam feed
    placeholder = st.empty()
 
    #try:
    faceid = True
    '''if start_button:
        da.read_video()
        print('function faceid init')
        faceid = da.faceid_init()
        print('function faceid finish')
        print('faceid: ', faceid)'''
        
    while start_button:# and faceid == False:
        # Capturar el video en tiempo real
        da.read_video()

        if faceid==False:
            print('begin with predictions')
            da.prediction()  
        else:
            print('function faceid init')
            faceid = da.faceid_init()
            print('function faceid finish')
            print('faceid: ', faceid)   

        # Mostrar el video
        placeholder.image(da.video_show(), channels='RGB')

        # Salir del bucle con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #finally:
    # Liberar la cámara y cerrar las ventanas
    da.finish()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()