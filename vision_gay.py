import cv2
from ultralytics import YOLO

# 学習済みのモデル（YOLOv8 Nano）を読み込みます
# 初回実行時に自動的にモデルファイル（yolov8n.pt）がダウンロードされます
model = YOLO('yolov8n.pt')

# Webカメラ（カメラ番号0）をオープンします
cap = cv2.VideoCapture(0)

print("認識を開始します。終了するには 'q' キーを押してください。")

while cap.isOpened():
    # カメラから1フレーム読み込みます
    success, frame = cap.read()

    if success:
        # モデルを使って物体検出を実行します
        # stream=Trueにすることでメモリ効率を上げ、リアルタイム処理をスムーズにします
        results = model(frame, stream=True)

        # 検出結果をフレームに描き込みます
        for r in results:
            annotated_frame = r.plot()

        # 結果を表示します
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # 'q'キーが押されたらループを抜けます
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 後処理
cap.release()
cv2.destroyAllWindows()