import cv2
import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
import pandas as pd

# 전역 변수로 클릭한 좌표 저장
original_coordinates = []

# 좌표 변환 함수를 정의
def transform_coordinates(coordinates, transformation, img_size):
    x, y = coordinates
    if transformation == "flipped_lr":
        x = img_size[0] - x  # 좌우 반전
    elif transformation == "flipped_ud":
        y = img_size[1] - y  # 상하 반전
    elif transformation == "flipped_both":
        x = img_size[0] - x  # 좌우 + 상하 반전
        y = img_size[1] - y
    elif transformation == "rotated_90":
        x, y = y, img_size[0] - x  # 90도 회전
    elif transformation == "rotated_180":
        x = img_size[0] - x  # 180도 회전
        y = img_size[1] - y
    elif transformation == "rotated_270":
        x, y = img_size[1] - y, x  # 270도 회전
    return x, y

# 파일명에서 확장자를 분리하고 증강된 파일 이름을 생성
def get_augmented_filename(filename, idx):
    name, ext = os.path.splitext(filename) # 파일 이름과 확장자 분리
    return f"{name}_aug{idx}{ext}" # 확장자 포함해서 증강된 파일 이름 생성

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

# 이미지를 선택하고 좌표를 저장하는 GUI 함수
def label_images():
    global original_coordinates

    # Tkinter 파일 다이얼로그로 이미지 폴더 선택
    root = tk.Tk()
    root.withdraw()  # Tkinter 윈도우 숨김
    folder_selected = filedialog.askdirectory(title="Select Image Folder")

    # 결과를 저장할 CSV 파일 경로
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Save Labels As")

    # 이미지를 불러오고 라벨링 시작
    image_files = [f for f in os.listdir(folder_selected) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)  # 총 이미지 수 계산
    labeled_data = []
    count = 0  # 진행률 카운터

    for image_file in image_files:
        # 이미지 경로 설정
        img_path = os.path.join(folder_selected, image_file)

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        # 사용자에게 클릭으로 좌표 선택
        img = cv2.resize(img, (256, 256))
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", lambda event, x, y, flags, param: save_click(event, x, y, param), img)

        print(f"Click on the image to select a point for {image_file}. Press any key after clicking.")
        cv2.waitKey(0)
        original_coordinates = (click_coordinates[0] if click_coordinates else (0, 0))  # 클릭한 좌표 저장
        cv2.destroyAllWindows()

        # 증강된 이미지 생성
        augmented_images, transformations = augment_image(img)

        for i, (aug_img, transformation) in enumerate(zip(augmented_images, transformations)):
            # 증강된 이미지로부터 새로운 좌표 계산
            new_coords = transform_coordinates(original_coordinates, transformation, img.shape[:2])
            
            # 증강된 이미지 저장
            aug_img_filename = get_augmented_filename(image_file, i)
            cv2.imwrite(os.path.join(folder_selected, aug_img_filename), aug_img)

            # 좌표 저장
            labeled_data.append([aug_img_filename, new_coords[0], new_coords[1]])
            print(f"Coordinates for {aug_img_filename}: {new_coords}")

            # 진행률 계산 및 출력
            count += 1
            progress = (count / (total_images * len(augmented_images))) * 100
            print(f"Progress: {progress:.2f}% ({count}/{total_images * len(augmented_images)} images)")

    # CSV 파일로 라벨 저장
    df = pd.DataFrame(labeled_data, columns=["image", "x", "y"])
    df.to_csv(save_path, index=False)
    print(f"Labels saved to {save_path}")

    cv2.destroyAllWindows()


# 클릭 이벤트 함수 (최종 클릭된 좌표만 저장)
def save_click(event, x, y, param):
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates = [(x, y)]
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", param)


if __name__ == "__main__":
    label_images()