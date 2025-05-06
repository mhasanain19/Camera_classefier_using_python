import tkinter as tk
import cv2
import os
import threading
from model import ImageClassifier
from PIL import Image, ImageTk

class ImageClassifierApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Image Classifier")
        self.classname_one = tk.StringVar()
        self.classname_two = tk.StringVar()
        self.classifier = ImageClassifier()

        self.label1 = tk.Label(window, text="Enter Class 1 Name:")
        self.label1.pack()
        self.entry1 = tk.Entry(window, textvariable=self.classname_one)
        self.entry1.pack()

        self.label2 = tk.Label(window, text="Enter Class 2 Name:")
        self.label2.pack()
        self.entry2 = tk.Entry(window, textvariable=self.classname_two)
        self.entry2.pack()

        self.start_button = tk.Button(window, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=10)

        self.status_label = tk.Label(window, text="", font=("Arial", 16))
        self.status_label.pack()

        self.video_label = tk.Label(window)
        self.video_label.pack()

        self.class_dirs = []

    def start_training(self):
        name1 = self.classname_one.get().strip()
        name2 = self.classname_two.get().strip()
        if not name1 or not name2:
            self.status_label.config(text="Please enter names for both classes.")
            return

        self.class_dirs = [f"data/{name1}", f"data/{name2}"]
        for directory in self.class_dirs:
            os.makedirs(directory, exist_ok=True)

        threading.Thread(target=self.capture_training_images).start()

    def capture_training_images(self):
        cap = cv2.VideoCapture(0)
        self.status_label.config(text="Capturing training images. Press '1' or '2' to capture, 'q' to finish.")
        count1 = count2 = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (640, 480))
            cv2.putText(frame_resized, "Press 1 for class 1, 2 for class 2, q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Capture Training Images", frame_resized)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                cv2.imwrite(f"{self.class_dirs[0]}/img_{count1}.jpg", frame)
                count1 += 1
            elif key == ord('2'):
                cv2.imwrite(f"{self.class_dirs[1]}/img_{count2}.jpg", frame)
                count2 += 1
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_label.config(text="Training the model...")
        self.classifier.train_model(self.class_dirs)
        self.status_label.config(text="Model trained! Starting real-time detection.")
        self.start_real_time_detection()

    def start_real_time_detection(self):
        def detect():
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                prediction = self.classifier.predict(frame)
                cv2.putText(frame, f"Detected: {prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                cv2.imshow("Real-Time Classification", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=detect).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
