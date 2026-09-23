import os
import time
import urllib.request
from typing import List, Tuple
import cv2
import numpy as np

class SelfContainedObjectDetector:

    PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    CAFFEMODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    PROTOTXT_PATH = "deploy.prototxt"
    CAFFEMODEL_PATH = "mobilenet_iter_73000.caffemodel"
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", 
        "sofa", "train", "tvmonitor"
    ]
    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self._ensure_model_files()
        self.net = cv2.dnn.readNetFromCaffe(self.PROTOTXT_PATH, self.CAFFEMODEL_PATH)

    def _ensure_model_files(self) -> None:
        if not os.path.exists(self.PROTOTXT_PATH):
            print("[INFO] Baixando arquivo de configuração (deploy.prototxt)...")
            urllib.request.urlretrieve(self.PROTOTXT_URL, self.PROTOTXT_PATH)

        if not os.path.exists(self.CAFFEMODEL_PATH):
            print("[INFO] Baixando pesos do modelo (caffemodel - aprox. 23MB)...")
            print("[INFO] Por favor, aguarde alguns segundos...")
            urllib.request.urlretrieve(self.CAFFEMODEL_URL, self.CAFFEMODEL_PATH)
            print("[INFO] Download concluído com sucesso!")

    def detect(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence > self.conf_threshold:
                class_id = int(detections[0, 0, i, 1])
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x_max, y_max = box.astype("int")
                x, y = max(0, x), max(0, y)
                w_box, h_box = min(w, x_max) - x, min(h, y_max) - y
                class_name = self.CLASSES[class_id] if class_id < len(self.CLASSES) else "desconhecido"
                results.append((class_name, confidence, (x, y, w_box, h_box)))

        return results

    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        annotated_frame = frame.copy()

        for class_name, confidence, (x, y, w, h) in detections:
            color = (0, 255, 0)  
            label = f"{class_name}: {confidence * 100:.1f}%"
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x, y - text_h - 6), (x + text_w, y), color, -1)
            
            cv2.putText(
                annotated_frame, label, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return annotated_frame

def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    detector = SelfContainedObjectDetector(conf_threshold=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a webcam.")
        return

    print("Pressione 's' para salvar um print da tela.")
    print("Pressione 'q' para sair.")
    
    prev_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Falha ao capturar frame da webcam.")
                break

            detections = detector.detect(frame)
            output_frame = detector.draw_detections(frame, detections)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            cv2.putText(
                output_frame, f"FPS: {fps:.1f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

            cv2.imshow("Detector de Objetos", output_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
                cv2.imwrite(filename, output_frame)
                print(f"Print salvo em: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Câmera fechada e recursos liberados.")

if __name__ == "__main__":
    main()
