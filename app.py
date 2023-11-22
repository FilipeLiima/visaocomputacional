import cv2
import numpy as np

# Carregando o modelo YOLOv3-Tiny
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Carregamento do vídeo
video_path = (
    "assets/people (6).mp4"  # Substitua com o caminho para o seu arquivo de vídeo
)
cap = cv2.VideoCapture(video_path)

# Define a largura desejada para o vídeo
desired_width = 840

# Loop para processar cada frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Falha ao ler o frame.")
        break

    height, width, _ = frame.shape

    # Calcula a nova altura proporcional à largura desejada
    desired_height = int((desired_width / width) * height)

    # Redimensiona o frame para a largura desejada
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Converte o frame para o formato que o YOLO espera
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(layer_names)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                detected_class = classes[class_id]

                if detected_class in ["person", "car"]:
                    # Desenha o retângulo ao redor do objeto detectado
                    center_x = int(detection[0] * desired_width)
                    center_y = int(detection[1] * desired_height)
                    w = int(detection[2] * desired_width)
                    h = int(detection[3] * desired_height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Adiciona o nome da classe próximo ao retângulo
                    label = f"{detected_class.capitalize()}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

    # Exibição do vídeo com as identificações
    cv2.imshow("Video com Identificação", frame)

    # Verifica se a tecla 'q' foi pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
