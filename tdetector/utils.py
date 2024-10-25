import cv2

def draw_boxes(frame, boxes, scores, classes, infractions, plate_recognizer):
    for box, score, cls in zip(boxes, scores, classes):
        if cls == 'car':
            color = (0, 255, 0)
            text = f'Car: {score}'
            if any(infraction[0] == box for infraction in infractions):
                color = (0, 0, 255)
                text += ', Infraction: Prohibited Zone'
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            plate_text = plate_recognizer.recognize_plate(frame[box[1]:box[3], box[0]:box[2]])
            cv2.putText(frame, f'{text}, Plate: {plate_text}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
