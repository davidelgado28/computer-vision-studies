"""
Extensão do main.py: reconhecimento de ALMOFADA.

Este arquivo NÃO altera o main.py. Ele importa o detector original
(SelfContainedObjectDetector) e adiciona, por cima, o reconhecimento da almofada
por foto de referência. Coloque os dois arquivos na mesma pasta e execute:

    python detector_almofada.py

Por que é necessário? O modelo MobileNet-SSD do main.py só conhece 20 classes
(pessoa, garrafa, sofá, cadeira...) e não possui "almofada".
Aqui comparamos a câmera com fotos de referência (SIFT + cor) e mostramos a
porcentagem de correspondência.
"""
import glob
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from main import SelfContainedObjectDetector

Detection = Tuple[str, float, Tuple[int, int, int, int]]

CLASS_NAME = "almofada"
CLASS_COLOR = (0, 165, 255)   # cor da caixa (BGR)


class DetectorComReferencias(SelfContainedObjectDetector):
    """Mesmo detector do main.py, só muda a cor da caixa da almofada."""

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_frame = frame.copy()

        for class_name, confidence, (x, y, w, h) in detections:
            color = CLASS_COLOR if class_name == CLASS_NAME else (0, 255, 0)
            label = f"{class_name}: {confidence * 100:.1f}%"
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x, y - text_h - 6), (x + text_w, y), color, -1)

            cv2.putText(
                annotated_frame, label, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return annotated_frame


class ReferenceMatcher:
    """
    Reconhece a almofada comparando a imagem da câmera com as fotos
    guardadas em reference/almofada/.

    A porcentagem final combina:
      - quantos pontos da foto de referência foram achados no frame (70%)
      - similaridade de cor entre a região achada e a referência (30%)
    """

    REFERENCE_DIR = os.path.join("reference", CLASS_NAME)
    CLASS_NAME = CLASS_NAME
    RATIO_TEST = 0.75          # filtro de Lowe: quanto menor, mais rigoroso
    MIN_INLIERS = 10           # mínimo de pontos coerentes para considerar encontrado
    FULL_SCORE_INLIERS = 40    # com esse número de pontos, a parte de formato vale 100%
    FEATURE_WEIGHT = 0.7
    COLOR_WEIGHT = 0.3

    def __init__(self, match_threshold: float = 0.30):
        self.match_threshold = match_threshold
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        # Cada referência: (imagem BGR, keypoints, descritores, histograma HSV)
        self.references = []
        os.makedirs(self.REFERENCE_DIR, exist_ok=True)
        self.load_references()

    # ---------- referências ----------
    @staticmethod
    def _color_hist(img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _add_reference(self, img: np.ndarray, name: str) -> bool:
        # Reduz fotos muito grandes para o processamento ficar rápido
        h, w = img.shape[:2]
        scale = 640.0 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < self.MIN_INLIERS:
            print(f"[AVISO] Pouca textura na referência '{name}'. Use uma foto mais nítida/próxima.")
            return False
        self.references.append((img, kp, des, self._color_hist(img)))
        print(f"[INFO] Referência carregada: {name} ({len(kp)} pontos)")
        return True

    def load_references(self) -> None:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            paths.extend(glob.glob(os.path.join(self.REFERENCE_DIR, ext)))
        for path in sorted(paths):
            img = cv2.imread(path)
            if img is not None:
                self._add_reference(img, os.path.basename(path))
        if not self.references:
            print(f"[AVISO] Nenhuma foto em '{self.REFERENCE_DIR}/'.")
            print("        Pressione 'r' durante a execução para cadastrar a almofada pela câmera.")

    def register_from_frame(self, frame: np.ndarray) -> bool:
        """Deixa o usuário desenhar um retângulo sobre o objeto e salva como referência."""
        print("[INFO] Desenhe um retângulo sobre a almofada e pressione ENTER (ESC cancela).")
        window = "Cadastrar almofada"
        x, y, w, h = cv2.selectROI(window, frame, showCrosshair=False)
        cv2.destroyWindow(window)
        if w < 20 or h < 20:
            print("[INFO] Cadastro cancelado.")
            return False
        crop = frame[y:y + h, x:x + w].copy()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.REFERENCE_DIR, f"{self.CLASS_NAME}_{timestamp}.jpg")
        cv2.imwrite(path, crop)
        print(f"[INFO] Referência salva em: {path}")
        return self._add_reference(crop, os.path.basename(path))

    # ---------- detecção ----------
    def _match_one(self, ref, frame: np.ndarray, kp_f, des_f) -> Optional[Detection]:
        ref_img, kp_r, des_r, ref_hist = ref
        h, w = frame.shape[:2]

        pairs = self.matcher.knnMatch(des_r, des_f, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.RATIO_TEST * p[1].distance]
        if len(good) < self.MIN_INLIERS:
            return None

        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None
        inliers = int(mask.sum())
        if inliers < self.MIN_INLIERS:
            return None

        # Projeta o contorno da referência no frame para achar onde o objeto está
        rh, rw = ref_img.shape[:2]
        corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        if not cv2.isContourConvex(proj.astype(np.int32)):
            return None  # formato distorcido demais = provável falso positivo

        x, y, bw, bh = cv2.boundingRect(proj.astype(np.int32))
        x, y = max(0, x), max(0, y)
        bw, bh = min(w, x + bw) - x, min(h, y + bh) - y
        if bw < 20 or bh < 20 or bw * bh > 0.95 * w * h:
            return None

        feature_score = min(1.0, inliers / self.FULL_SCORE_INLIERS)
        region = frame[y:y + bh, x:x + bw]
        color_score = max(0.0, cv2.compareHist(ref_hist, self._color_hist(region), cv2.HISTCMP_CORREL))
        confidence = self.FEATURE_WEIGHT * feature_score + self.COLOR_WEIGHT * color_score
        return (self.CLASS_NAME, float(confidence), (x, y, bw, bh))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.references:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = self.sift.detectAndCompute(gray, None)
        if des_f is None or len(kp_f) < self.MIN_INLIERS:
            return []

        best = None
        for ref in self.references:
            result = self._match_one(ref, frame, kp_f, des_f)
            if result and (best is None or result[1] > best[1]):
                best = result

        if best and best[1] >= self.match_threshold:
            return [best]
        return []


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    detector = DetectorComReferencias(conf_threshold=0.5)
    matcher = ReferenceMatcher(match_threshold=0.30)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a webcam.")
        return

    print("Pressione 'r' para cadastrar a almofada (desenhe um retângulo sobre ela).")
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
            detections += matcher.detect(frame)
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

            elif key == ord('r'):
                matcher.register_from_frame(frame)
                prev_time = 0.0

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
