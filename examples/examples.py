import cv2
from tdetector import VehicleDetector, PlateRecognizer, InfractionChecker, utils

# Configuración
model_path = 'yolo_model.h5'
# tesseract.exe address
tesseract_cmd = r'C:\Users\M-Castro\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Inicialización
vehicle_detector = VehicleDetector(model_path)
plate_recognizer = PlateRecognizer(tesseract_cmd)
infraction_checker = InfractionChecker()

# Cargar video
cap = cv2.VideoCapture('example_video.mp4')

# Abrir archivo de texto para escribir
with open('infractions.txt', 'w') as file:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar objetos
        predictions = vehicle_detector.detect_objects(frame)
        boxes, scores, classes = vehicle_detector.process_predictions(predictions)

        # Detectar infracciones
        infractions = infraction_checker.detect_infractions(frame, boxes, scores, classes)

        # Dibujar cajas delimitadoras y reconocer placas
        utils.draw_boxes(frame, boxes, scores, classes, infractions, plate_recognizer)

        # Guardar infracciones en el archivo de texto
        for infraction in infractions:
            box, score, cls, infraction_type = infraction
            plate_text = plate_recognizer.recognize_plate(frame[box[1]:box[3], box[0]:box[2]])
            file.write(f'Plate: {plate_text}, Infraction: {infraction_type}\n')

        cv2.imshow('Detections', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
