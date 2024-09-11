import cv2
import os

import keras
import pandas as pd
import numpy as np
import tensorflow as tf

from keras import layers, models
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau


# 모델 정의 (이미지 -> 카메라 각도 예측 모델)
def build_camera_control_model(input_shape=(128, 128, 1)):
    # 입력 레이어
    inputs = layers.Input(shape=input_shape)

    # 첫 번째 합성곱 층 + 맥스풀링 + Batch Normalization
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001),
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 두 번째 합성곱 층 + 맥스풀링 + Batch Normalization
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 세 번째 합성곱 층 + 맥스풀링 + Batch Normalization
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 네 번째 합성곱 층 + 맥스풀링 + Batch Normalization (층을 더 깊게 구성)
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.swish(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 평탄화 (Flatten) 후 Fully Connected 층
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001))(x)  # 노드를 512로 증가
    x = keras.activations.swish(x)
    x = layers.Dropout(0.3)(x)  # Dropout 추가

    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001))(x)
    x = keras.activations.swish(x)
    x = layers.Dropout(0.3)(x)  # Dropout 추가

    x = layers.Dense(32, kernel_regularizer=keras.regularizers.l1_l2(0.001, 0.001))(x)
    x = keras.activations.swish(x)
    x = layers.Dropout(0.3)(x)  # Dropout 추가

    # 출력층 (카메라 상하 및 좌우 각도 예측)
    outputs = layers.Dense(2, activation='linear')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# CSV 파일에서 라벨링된 데이터를 로드하는 함수
def load_labeled_data(csv_path, image_folder, empty_folder, img_size=(128, 128)):
    labeled_data = pd.read_csv(csv_path)

    images = []
    labels = []

    # 공이 있는 이미지와 좌표 불러오기
    for _, row in labeled_data.iterrows():
        image_file = row['image']
        x, y = row['x'], row['y']

        img_path = str(os.path.join(image_folder, image_file))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=-1)  # (128, 128, 1)

        # 좌표 정규화 (0~128 사이 값을 -1~1 사이로 변환)
        norm_x = (x / img_size[0]) * 2 - 1
        norm_y = (y / img_size[1]) * 2 - 1

        images.append(img)
        labels.append([norm_x, norm_y])

    # 공이 없는 이미지에 대해서는 (0, 0)을 라벨로 할당
    for empty_file in os.listdir(empty_folder):
        empty_img_path = str(os.path.join(empty_folder, empty_file))
        img = cv2.imread(empty_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to load image: {empty_img_path}")
            continue

        img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=-1)  # (128, 128, 1)

        images.append(img)
        labels.append([0, 0])  # 공이 없을 때는 (0, 0) 출력

    return np.array(images), np.array(labels)


# 데이터셋 로드 함수
def load_dataset(train_csv, train_folder, empty_folder, test_csv, test_folder, test_empty_folder, img_size=(128, 128)):
    # 학습 데이터 로드 (공이 있는 데이터와 없는 데이터 함께)
    train_images, train_labels = load_labeled_data(train_csv, train_folder, empty_folder, img_size)

    # 테스트 데이터 로드 (공이 있는 데이터만)
    test_images, test_labels = load_labeled_data(test_csv, test_folder, test_empty_folder, img_size)

    return train_images, train_labels, test_images, test_labels


# 테스트 데이터에 대해 예측을 수행하고, 그림을 저장하는 함수
def test_and_save_results(model, test_images, test_labels, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)

    # 테스트 데이터에 대해 예측 수행
    predictions = model.predict(test_images)

    for i, (image, predicted_action, target_action) in enumerate(zip(test_images, predictions, test_labels)):
        # 결과 이미지를 저장
        save_result_image(image, predicted_action, target_action, i + 1, save_dir)

    print(f"All results saved to {save_dir}")

    # TensorFlow Lite Converter 사용
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 양자화 옵션 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 대표 데이터셋 설정 (float32 -> int8 양자화에 사용)
    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):  # 데이터셋의 일부를 사용
            yield [input_value.numpy().astype(np.float32)]  # float32로 입력

    # 대표 데이터셋을 양자화에 사용
    converter.representative_dataset = representative_data_gen

    # 입력 및 출력 타입을 uint8로 설정 (EdgeTPU에서 필요)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # 입력을 uint8로 설정
    converter.inference_output_type = tf.uint8  # 출력을 uint8로 설정

    # 모델을 TensorFlow Lite로 변환
    tflite_model = converter.convert()

    # 변환된 모델 저장
    with open('model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)

    print("TensorFlow Lite model has been saved as 'model_quantized.tflite'")


# 학습 함수
def train_camera_control_model(train_images, train_labels, test_images, test_labels,
                               model_save_path="./models/final_model.h5", batch_size=32, epochs=500,
                               results_dir="./results"):
    model = build_camera_control_model()

    # 모델 컴파일
    model.compile(optimizer=keras.optimizers.AdamW(1e-4, weight_decay=1e-5), loss='mean_squared_error', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 학습
    history = model.fit(
        train_images, train_labels, validation_data=(test_images, test_labels),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr],
        shuffle=True
    )

    # 학습 완료 후 모델 저장
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Final model saved to {model_save_path}")

    # 테스트 데이터에 대해 예측을 수행하고 결과 저장
    test_and_save_results(model, test_images, test_labels, save_dir=results_dir)

    return model, history


# 이미지에 예측된 위치와 목표 위치를 그려서 저장하는 함수
def save_result_image(state, predicted_action, target_action, episode, save_dir):
    # 이미지를 (128, 128, 1) -> (128, 128)로 변환
    img = state.squeeze()

    # 그레이스케일 이미지를 컬러 이미지로 변환 (BGR 채널로)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 예측된 좌표와 목표 좌표를 실제 이미지 크기에 맞게 변환 ([-1, 1] 범위를 [0, 128]로 변환)
    predicted_x = int((predicted_action[0] + 1) * 64)
    predicted_y = int((predicted_action[1] + 1) * 64)
    target_x = int((target_action[0] + 1) * 64)
    target_y = int((target_action[1] + 1) * 64)

    # 예측된 위치를 빨간색 원으로 표시
    cv2.circle(img_color, (predicted_x, predicted_y), 5, (0, 0, 255), -1)  # 빨간색 원: 예측 위치

    # 목표 위치를 초록색 원으로 표시
    cv2.circle(img_color, (target_x, target_y), 5, (0, 255, 0), -1)  # 초록색 원: 목표 위치

    # 이미지 파일로 저장
    image_path = os.path.join(save_dir, f"episode_{episode}.png")
    cv2.imwrite(image_path, img_color)
    print(f"Saved image for episode {episode} to {image_path}")


# 메인 실행 함수
if __name__ == '__main__':
    # 학습 및 테스트 데이터 경로 설정
    train_csv = './data/train_labels.csv'
    train_folder = './data/train/augmented'
    empty_folder = './data/train/empty'
    test_csv = './data/test_labels.csv'
    test_folder = './data/test/augmented'
    test_empty_folder = './data/test/empty'

    # 데이터셋 로드
    train_images, train_labels, test_images, test_labels = load_dataset(train_csv, train_folder, empty_folder, test_csv,
                                                                        test_folder, test_empty_folder)

    print(f"Loaded: {len(train_images)} training images, {len(test_images)} testing images")

    # 모델 학습
    print("Starting training...")
    model, history = train_camera_control_model(train_images, train_labels, test_images, test_labels)

    print("Training complete!")
