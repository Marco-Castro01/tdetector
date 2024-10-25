import unittest
import cv2
from tensorflow.python.ops.logging_ops import Print

from tdetector import PlateRecognizer

class TestPlateRecognizer(unittest.TestCase):
    def setUp(self):
        self.recognizer = PlateRecognizer(tesseract_cmd=r'C:\Users\M-Castro\AppData\Local\Programs\Tesseract-OCR\tesseract.exe')
        self.image = cv2.imread('img/img.png')

    def test_recognize_plate(self):
        plate_text = self.recognizer.recognize_plate(self.image)
        print(f"Plate text: {plate_text}")
        self.assertIsNotNone(plate_text)

if __name__ == '__main__':
    unittest.main()
