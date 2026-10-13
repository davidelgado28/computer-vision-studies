import os
import time
import urllib.request
from typing import List, Tuple
import cv2
import numpy as np

class DetectorObjetos:
    url_prototxt = "https://githubusercontent.com"
    url_caffemodel = "https://githubusercontent.com"
    caminho_prototxt = "deploy.prototxt"
    caminho_caffemodel = "mobilenet_iter_73000.caffemodel"
    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    def __init__(self, limite_confianca: float = 0.5):
        self.limite_confianca = limite_confianca
        np.random.seed(42)
        self.cores = np.random.uniform(0, 255, size=(len(self.classes), 3)).astype("int")
        self.garantir_arquivos_modelo()
        self.rede = cv2.dnn.readNetFromCaffe(self.caminho_prototxt, self.caminho_caffemodel)

    def garantir_arquivos_modelo(self) -> None:
        if not os.path.exists(self.caminho_prototxt):
            print("[INFO] Baixando arquivo de configuracao (deploy.prototxt)...")
            urllib.request.urlretrieve(self.url_prototxt, self.caminho_prototxt)
        if not os.path.exists(self.caminho_caffemodel):
            print("[INFO] Baixando pesos do modelo (caffemodel - aprox. 23MB)...")
            print("[INFO] Por favor, aguarde alguns segundos...")
            urllib.request.urlretrieve(self.url_caffemodel, self.caminho_caffemodel)
            print("[INFO] Download concluido com sucesso!")

    def detectar(self, quadro: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int], int]]:
        altura, largura = quadro.shape[:2]
        imagem_blob = cv2.dnn.blobFromImage(
            cv2.resize(quadro, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self.rede.setInput(imagem_blob)
        deteccoes = self.rede.forward()
        resultados = []
        
        for indice in range(deteccoes.shape[2]):
            confianca = float(deteccoes[0, 0, indice, 2])
            
            if confianca <= self.limite_confianca:
                continue
                
            id_classe = int(deteccoes[0, 0, indice, 1])
            coordenadas_caixa = deteccoes[0, 0, indice, 3:7] * np.array([largura, altura, largura, altura])
            coord_x, coord_y, x_max, y_max = coordenadas_caixa.astype("int")
            
            coord_x = max(0, coord_x)
            coord_y = max(0, coord_y)
            largura_caixa = min(largura, x_max) - coord_x
            altura_caixa = min(altura, y_max) - coord_y
            
            nome_classe = self.classes[id_classe] if id_classe < len(self.classes) else "desconhecido"
            resultados.append((nome_classe, confianca, (coord_x, coord_y, largura_caixa, altura_caixa), id_classe))
            
        return resultados

    def desenhar_deteccoes(self, quadro: np.ndarray, deteccoes: List[Tuple[str, float, Tuple[int, int, int, int], int]]) -> np.ndarray:
        quadro_anotado = quadro.copy()
        for nome_classe, confianca, (coord_x, coord_y, largura_caixa, altura_caixa), id_classe in deteccoes:
            cor = [int(c) for c in self.cores[id_classe]]
            rotulo = f"{nome_classe}: {confianca * 100:.1f}%"
            
            cv2.rectangle(quadro_anotado, (coord_x, coord_y), (coord_x + largura_caixa, coord_y + altura_caixa), cor, 2)
            (largura_texto, altura_texto), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(quadro_anotado, (coord_x, coord_y - altura_texto - 6), (coord_x + largura_texto, coord_y), cor, -1)
            cv2.putText(
                quadro_anotado, rotulo, (coord_x, coord_y - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        return quadro_anotado

def principal():
    diretorio_saida = "output"
    os.makedirs(diretorio_saida, exist_ok=True)
    
    detector = DetectorObjetos(limite_confianca=0.5)
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("[ERRO] Nao foi possivel acessar a webcam.")
        return
        
    print("Pressione 's' para salvar um print da tela.")
    print("Pressione 'q' para sair.")
    tempo_anterior = 0.0
    
    try:
        while True:
            sucesso, quadro = camera.read()
            if not sucesso:
                print("[AVISO] Falha ao capturar frame da webcam.")
                break
                
            deteccoes = detector.detectar(quadro)
            quadro_saida = detector.desenhar_deteccoes(quadro, deteccoes)
            
            tempo_atual = time.time()
            fps = 1.0 / (tempo_atual - tempo_anterior) if tempo_anterior > 0 else 0.0
            tempo_anterior = tempo_atual
            
            cv2.putText(
                quadro_saida, f"FPS: {fps:.1f}", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            
            cv2.imshow("Detector de Objetos", quadro_saida)
            tecla_pressionada = cv2.waitKey(1) & 0xFF
            
            if tecla_pressionada == ord('q'):
                break
            elif tecla_pressionada == ord('s'):
                marca_tempo = time.strftime("%Y%m%d%H%M%S")
                nome_arquivo_foto = os.path.join(diretorio_saida, f"snapshot_{marca_tempo}.jpg")
                cv2.imwrite(nome_arquivo_foto, quadro_saida)
                print(f"Print salvo em: {nome_arquivo_foto}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera fechada e recursos liberados.")

if __name__ == "__main__":
    principal()
