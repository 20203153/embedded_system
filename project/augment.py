import cv2
import os
import random
import numpy as np


# 이미지 증강 함수
def augment_image(image, image_name, save_dir, num_replica=5):
    # 적용할 증강 기법들
    transformations = [
        ('rotate_90', lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ('rotate_180', lambda img: cv2.rotate(img, cv2.ROTATE_180)),
        ('rotate_270', lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ('flip_horizontal', lambda img: cv2.flip(img, 1)),
        ('flip_vertical', lambda img: cv2.flip(img, 0)),
        ('flip_both', lambda img: cv2.flip(img, -1)),
        ('shift_left', lambda img: shift_image(img, shift_x=-10)),  # 왼쪽으로 이동
        ('shift_right', lambda img: shift_image(img, shift_x=10)),  # 오른쪽으로 이동
        ('shift_up', lambda img: shift_image(img, shift_y=-10)),  # 위로 이동
        ('shift_down', lambda img: shift_image(img, shift_y=10)),  # 아래로 이동
        ('scale_crop', lambda img: scale_and_crop(img, scale=1.2)),  # 크기 조정 후 자르기
        ('brightness_increase', lambda img: adjust_brightness(img, 1.2)),  # 밝기 증가
        ('brightness_decrease', lambda img: adjust_brightness(img, 0.8)),  # 밝기 감소
        ('add_noise', lambda img: add_gaussian_noise(img)),  # 가우시안 노이즈 추가
    ]
    # 최소 5개, 최대 9개 랜덤으로 선택
    selected_transformations = random.sample(transformations, random.randint(5, 9))

    # 증강된 이미지를 저장
    for i, (trans_name, transform_func) in enumerate(selected_transformations):
        augmented_img = transform_func(image)

        # 저장할 파일 이름 생성
        replica_name = f"{image_name}_rep{i+1}.jpg"
        save_path = os.path.join(save_dir, replica_name)

        # 증강된 이미지 저장
        cv2.imwrite(save_path, augmented_img)
        print(f"Saved augmented image: {save_path}")


# 이미지 이동 (좌우/상하 이동)
def shift_image(image, shift_x=0, shift_y=0):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted


# 크기 조정 및 자르기
def scale_and_crop(image, scale=1.2):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # 중앙 자르기
    start_x = new_w // 2 - w // 2
    start_y = new_h // 2 - h // 2
    cropped = resized[start_y:start_y + h, start_x:start_x + w]
    return cropped


# 밝기 조정
def adjust_brightness(image, factor=1.0):
    image = image.astype(np.float32)  # 연산을 위해 float32로 변환
    adjusted = np.clip(image * factor, 0, 255)  # 밝기 조정 후 클리핑
    return adjusted.astype(np.uint8)


# 가우시안 노이즈 추가
def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
    return noisy_image


# 디렉토리에서 이미지를 불러와서 증강하는 함수
def augment_images_in_folder(folder_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 폴더 내의 모든 이미지 파일 불러오기
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, image_file)

            # 이미지 로드
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Unable to load image: {img_path}")
                continue

            # 이미지 이름에서 확장자를 제거하여 기본 이름 추출
            image_name, _ = os.path.splitext(image_file)

            # 최소 5개, 최대 9개의 증강된 이미지를 저장
            augment_image(img, image_name, save_dir)


# 실행 예시
if __name__ == "__main__":
    folder_path = './data/train/balls'  # 원본 이미지 폴더
    save_dir = './data/train/augmented/balls'  # 증강된 이미지를 저장할 폴더

    folder_test_path = './data/test/balls'
    save_test_dir = './data/test/augmented/balls'

    augment_images_in_folder(folder_path, save_dir)
    augment_images_in_folder(folder_test_path, save_test_dir)
