import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

class ImageClassifier:
    def __init__(self):
        self.model = LinearSVC()
        self.labels = []
        self.encoder = LabelEncoder()

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (150, 150))
        return resized.flatten()

    def train_model(self, data_dirs):
        X = []
        y = []
        for dir_path in data_dirs:
            class_name = os.path.basename(dir_path)
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                features = self.extract_features(image)
                X.append(features)
                y.append(class_name)
        if not X:
            raise Exception("No training data found!")
        self.labels = list(set(y))
        y_encoded = self.encoder.fit_transform(y)
        self.model.fit(X, y_encoded)

    def predict(self, frame):
        features = self.extract_features(frame)
        prediction_encoded = self.model.predict([features])[0]
        return self.encoder.inverse_transform([prediction_encoded])[0]
