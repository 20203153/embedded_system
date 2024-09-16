import cv2
import os
import tkinter as tk
import numpy as np
from tkinter import filedialog
import pandas as pd

# 전역 변수로 클릭한 좌표 저장
click_coordinates = []


# 클릭 이벤트 함수 (최종 클릭된 좌표만 저장)
def click_event(event, x, y, flags, param):
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        # 좌표를 저장 (여러 번 클릭할 때 마지막 클릭만 저장)
        click_coordinates = [(x, y)]
        print(f"Clicked at: ({x}, {y})")

        # 클릭한 위치에 원 그리기
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", param)


# 이미지 증강 함수 (모든 가능한 증강 적용)
def augment_image(image):
    augmented_images = []

    # 원본 이미지 추가
    augmented_images.append(image)

    # 1. 좌우 반전
    flipped_lr = cv2.flip(image, 1)
    augmented_images.append(flipped_lr)

    # 2. 상하 반전
    flipped_ud = cv2.flip(image, 0)
    augmented_images.append(flipped_ud)

    # 3. 좌우 + 상하 반전
    flipped_both = cv2.flip(image, -1)
    augmented_images.append(flipped_both)

    # 4. 회전 (90도, 180도, 270도)
    for angle in [90, 180, 270]:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if angle == 90 else
                             (cv2.ROTATE_180 if angle == 180 else cv2.ROTATE_90_COUNTERCLOCKWISE))
        augmented_images.append(rotated)

    return augmented_images


# 파일명에서 확장자를 분리하고 증강된 파일 이름을 생성
def get_augmented_filename(filename, idx):
    name, ext = os.path.splitext(filename)  # 파일 이름과 확장자 분리
    return f"{name}_aug{idx}{ext}"  # 확장자 포함해서 증강된 파일 이름 생성


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
    total_images = len(image_files)  # 총 이미지 수 계산
    labeled_data = []
    count = 0  # 진행률 카운터

    for image_file in image_files:
        # 이미지 경로 설정
        img_path = os.path.join(folder_selected, image_file)

        # 이미지 로드 (128x128 컬러)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        # 이미지 크기를 128x128로 변환
        img = cv2.resize(img, (128, 128))

        # 증강된 이미지 생성
        augmented_images = augment_image(img)

        for i, aug_img in enumerate(augmented_images):
            # 증강된 이미지 저장
            aug_img_filename = get_augmented_filename(image_file, i)
            cv2.imwrite(os.path.join(folder_selected, aug_img_filename), aug_img)

            # 이미지 윈도우 생성 및 이벤트 설정
            cv2.imshow("Image", aug_img)
            cv2.setMouseCallback("Image", click_event, aug_img)

            print(f"Label the augmented image: {aug_img_filename}")
            click_coordinates = []  # 클릭 좌표 초기화

            # 's' 키를 눌러 저장, 'n' 키를 눌러 건너뜀
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s') and click_coordinates:
                    # 좌표를 CSV로 저장 (증강된 이미지와 함께)
                    labeled_data.append([aug_img_filename, click_coordinates[0][0], click_coordinates[0][1]])
                    print(f"Coordinates saved for {aug_img_filename}: {click_coordinates[0]}")
                    break
                elif key == ord('n'):
                    print(f"Skipped {aug_img_filename}")
                    break

            # 진행률 계산 및 출력
            count += 1
            progress = (count / (total_images * len(augmented_images))) * 100
            print(f"Progress: {progress:.2f}% ({count}/{total_images * len(augmented_images)} images)")

    # CSV 파일로 라벨 저장
    df = pd.DataFrame(labeled_data, columns=["image", "x", "y"])
    df.to_csv(save_path, index=False)
    print(f"Labels saved to {save_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    label_images()
