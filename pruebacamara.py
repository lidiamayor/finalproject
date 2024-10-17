import cv2

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break

    # Aquí puedes agregar el procesamiento de tu modelo, por ejemplo, detección con YOLO

    cv2.imshow('Detección de Fatiga', frame)

    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
