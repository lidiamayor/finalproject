import cv2

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break

    cv2.imshow('Deteccion de Fatiga', frame)

    # Presionar 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
