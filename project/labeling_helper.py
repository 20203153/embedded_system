import cv2
import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
import pandas as pd

# 전역 변수로 클릭한 좌표 저장
click_coordinates = []

# 클릭 이벤트 함수
def save_click_event(event, x, y, flags, param):
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates = [(x, y)]  # 버튼 클릭한 좌표 저장
        print(f"Clicked at: ({x}, {y})")

# 좌표 변환 함수 정의
def transform_coordinates(coordinates, transformation, img_size):
    x, y = coordinates
    height, width = img_size  # 이미지 크기 가져오기
    
    if transformation == "flipped_lr":  # 좌우 반전
        x = width - x
    elif transformation == "flipped_ud":  # 상하 반전
        y = height - y
    elif transformation == "flipped_both":  # 좌우 + 상하 반전
        x = width - x
        y = height - y
    elif transformation == "rotated_90":  # 90도 회전
        x, y = y, width - x
    elif transformation == "rotated_180":  # 180도 회전
        x = width - x
        y = height - y
    elif transformation == "rotated_270":  # 270도 회전
        x, y = height - y, x
    return int(min(max(x, 0), width-1)), int(min(max(y, 0), height-1))

# 이미지 증강 함수 (모든 가능한 증강 적용)
def augment_image(image):
    augmented_images = []
    transformations = []

    # 원본 이미지 추가
    augmented_images.append(image)
    transformations.append("original")

    # 1. 좌우 반전
    flipped_lr = cv2.flip(image, 1)
    augmented_images.append(flipped_lr)
    transformations.append("flipped_lr")

    # 2. 상하 반전
    flipped_ud = cv2.flip(image, 0)
    augmented_images.append(flipped_ud)
    transformations.append("flipped_ud")

    # 3. 좌우 + 상하 반전
    flipped_both = cv2.flip(image, -1)
    augmented_images.append(flipped_both)
    transformations.append("flipped_both")

    # 4. 회전 (90도, 180도, 270도)
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(rotated_90)
    transformations.append("rotated_90")

    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    augmented_images.append(rotated_180)
    transformations.append("rotated_180")

    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    augmented_images.append(rotated_270)
    transformations.append("rotated_270")

    return augmented_images, transformations

# 이미지를 선택하고 라벨을 저장하는 GUI 함수
def label_images(mode):
    # Tkinter 파일 다이얼로그로 이미지 폴더 선택
    root = tk.Tk()
    root.withdraw()  # Tkinter 윈도우 숨김
    folder_selected = filedialog.askdirectory(title="Select Image Folder")
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save Labels As")

    # 기존 CSV 파일이 있을 경우 데이터프레임으로 읽기
    if os.path.exists(save_path):
        existing_data = pd.read_csv(save_path)

    labeled_data = []
    image_files = [f for f in os.listdir(folder_selected) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)  # 총 이미지 수 계산
    count = 0  # 진행률 카운터

    for image_file in image_files:
        # 이미 증강된 파일인지 확인
        if "_aug" in image_file:
            print(f"Skipping already augmented image: {image_file}")
            continue

        img_path = os.path.join(folder_selected, image_file)

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        # 이미지 크기를 256x256으로 변환
        img = cv2.resize(img, (256, 256))
        original_coordinates = []

        while True:
            cv2.imshow("Image", img)
            global click_coordinates
            click_coordinates = []  # 클릭 좌표 초기화
            cv2.setMouseCallback("Image", save_click_event)

            print(f"Click on the image to select a point for {image_file}. Press 'S' to save, 'N' to skip.")

            # 루프를 통해 사용자 입력을 처리
            while True:
                key = cv2.waitKey(1)
                if key == ord('s'):
                    if click_coordinates:  # 좌표가 선택되었을 경우에만 저장
                        original_coordinates = click_coordinates[0]
                        break
                elif key == ord('n'):
                    print(f"Skipped {image_file}")
                    original_coordinates = None
                    break

            if original_coordinates is not None:
                # 증강된 이미지 생성 및 라벨링
                augmented_images, transformations = augment_image(img)

                for i, (aug_img, transformation) in enumerate(zip(augmented_images, transformations)):
                    filename = get_augmented_filename(image_file, i)
                    new_coords = transform_coordinates(original_coordinates, transformation, img.shape[:2])
                    labeled_data.append([filename, new_coords[0], new_coords[1]])

                    cv2.imwrite(os.path.join(folder_selected, filename), aug_img)

                    count += 1
                    progress = (count / (total_images * len(augmented_images))) * 100
                    print(f"Progress: {progress:.2f}% ({count}/{total_images * len(augmented_images)} images)")

            # 이미지 창을 닫고 다음 이미지로 넘어감
            cv2.destroyAllWindows()
            break

    # 새로운 데이터와 기존 데이터 통합
    if mode == "augment" and os.path.exists(save_path):
        combined_data = pd.concat([existing_data, pd.DataFrame(labeled_data, columns=["image", "x", "y"])])
    else:
        combined_data = pd.DataFrame(labeled_data, columns=["image", "x", "y"])

    combined_data.to_csv(save_path, index=False)
    print(f"Labels saved to {save_path}")

    cv2.destroyAllWindows()

# 증강된 파일 이름 생성 함수
def get_augmented_filename(filename, idx):
    name, ext = os.path.splitext(filename)
    return f"{name}_aug{idx}{ext}"

if __name__ == "__main__":
    mode = input("Choose mode (label/augment): ").strip().lower()
    if mode not in ["label", "augment"]:
        print("Invalid mode selected. Please use 'label' or 'augment'.")
    else:
        label_images(mode)
