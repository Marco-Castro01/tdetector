import cv2
import numpy as np
import tensorflow as tf

class VehicleDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def detect_objects(self, image):
        image_resized = cv2.resize(image, (416, 416))
        image_normalized = image_resized / 255.0
        image_expanded = np.expand_dims(image_normalized, axis=0)
        predictions = self.model.predict(image_expanded)
        return predictions

    def process_predictions(self, predictions):
        boxes = []
        scores = []
        classes = []
        for pred in predictions:
            boxes.append(pred[0:4])
            scores.append(pred[4])
            classes.append(pred[5])
        return boxes, scores, classes
