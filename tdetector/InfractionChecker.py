class InfractionChecker:
    def detect_infractions(self, frame, boxes, scores, classes):
        infractions = []
        for box, score, cls in zip(boxes, scores, classes):
            if cls == 'car':
                if box[0] < 100:
                    infractions.append((box, score, cls, 'Prohibited Zone'))
        return infractions
