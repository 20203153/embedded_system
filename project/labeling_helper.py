import cv2
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# 전역 변수로 클릭한 좌표 저장
click_coordinates = []

# 클릭 이벤트 함수
def click_event(event, x, y, flags, param):
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN and len(click_coordinates) == 0:
        # 좌표를 저장 (이미지 하나당 하나의 좌표만 허용)
        click_coordinates.append((x, y))
        print(f"Clicked at: ({x}, {y})")

        # 클릭한 위치에 원 그리기
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

# 이미지를 선택하고 좌표를 저장하는 GUI 함수
def label_images():
    global img, click_coordinates

    # Tkinter 파일 다이얼로그로 이미지 폴더 선택
    root = tk.Tk()
    root.withdraw()  # Tkinter 윈도우 숨김
    folder_selected = filedialog.askdirectory(title="Select Image Folder")

    # 결과를 저장할 CSV 파일 경로
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save Labels As")

    # 이미지를 불러오고 라벨링 시작
    image_files = [f for f in os.listdir(folder_selected) if f.endswith(('.png', '.jpg', '.jpeg'))]
    labeled_data = []

    for image_file in image_files:
        # 이미지 경로 설정
        img_path = os.path.join(folder_selected, image_file)

        # 이미지 로드 (128x128 그레이스케일)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        # 이미지 크기를 128x128로 변환
        img = cv2.resize(img, (128, 128))

        # 이미지 윈도우 생성 및 이벤트 설정
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", click_event)

        print(f"Label the image: {image_file}")
        click_coordinates = []  # 클릭 좌표 초기화

        # 's' 키를 눌러 저장, 'n' 키를 눌러 건너뜀
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s') and click_coordinates:
                # 좌표를 CSV로 저장
                labeled_data.append([image_file, click_coordinates[0][0], click_coordinates[0][1]])
                print(f"Coordinates saved for {image_file}: {click_coordinates[0]}")
                break
            elif key == ord('n'):
                print(f"Skipped {image_file}")
                break

    # CSV 파일로 라벨 저장
    df = pd.DataFrame(labeled_data, columns=["image", "x", "y"])
    df.to_csv(save_path, index=False)
    print(f"Labels saved to {save_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    label_images()
