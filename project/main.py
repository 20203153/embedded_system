import cv2
import glob
import os
import pandas as pd

import keras
from keras import layers
import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

def build_image_dqn__model(input_shape=(128, 128, 1)):
    # 입력 레이어
    inputs = layers.Input(shape=input_shape)

    # 첫 번째 합성곱 층 + 맥스풀링
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # 두 번째 합성곱 층 + 맥스풀링
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 세 번째 합성곱 층 + 맥스풀링
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 평탄화 (Flatten) 후 Fully Connected 층
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # 출력층 (좌우 및 상하 각도 예측)
    outputs = layers.Dense(2, activation='tanh')(x)

    # 입력과 출력을 정의하여 모델 생성
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


class ImageDQNAgent:
    def __init__(self, state_shape=(128, 128, 1), action_size=(2,)):
        self.state_shape = state_shape
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.model = build_image_dqn__model(self.state_shape)
        self.model.compile(optimizer=keras.optimizers.Adagrad(self.learning_rate), loss='mse', metrics=['accuracy'])

        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-1, 1, self.action_size)

        # state의 차원을 맞추기 위해 차원 추가 (배치 차원 추가)
        state = np.expand_dims(state, axis=0)  # (128, 128, 1) -> (1, 128, 128, 1)
        state = np.expand_dims(state, axis=-1) if state.ndim == 3 else state  # (128, 128) -> (128, 128, 1)
        act_values = self.model.predict(state)
        return act_values[0]

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        target = self.model.predict(states)
        target_next = self.model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                target[i] = rewards[i]
            else:
                target[i] = rewards[i] + self.gamma * np.max(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def calculate_direction_to_target(center, target=(64, 64)):
    delta_x = center[0] - target[0]
    delta_y = center[1] - target[1]

    move_x = delta_x / 64.0
    move_y = delta_y / 64.0

    return np.clip([move_x, move_y], -1, 1)


# CSV 파일에서 라벨링된 데이터를 로드하는 함수
def load_labeled_data(csv_path, image_folder, img_size=(128, 128)):
    labeled_data = pd.read_csv(csv_path)

    images = []
    labels = []

    for _, row in labeled_data.iterrows():
        # 이미지 파일 이름과 좌표 불러오기
        image_file = row['image']
        x, y = row['x'], row['y']

        # 이미지 로드
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Unable to load image: {img_path}")
            continue

        # 이미지 전처리 (크기 조정 및 차원 확장)
        img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=-1)  # (128, 128, 1)

        # 좌표 정규화 (0~128 사이 값을 -1~1 사이로 변환)
        norm_x = (x / img_size[0]) * 2 - 1
        norm_y = (y / img_size[1]) * 2 - 1

        images.append(img)
        labels.append([norm_x, norm_y])

    return np.array(images), np.array(labels)


# 데이터셋 로드 함수
def load_dataset(train_csv, train_folder, test_csv, test_folder, img_size=(128, 128)):
    # 학습 데이터 로드
    train_images, train_labels = load_labeled_data(train_csv, train_folder, img_size)

    # 테스트 데이터 로드
    test_images, test_labels = load_labeled_data(test_csv, test_folder, img_size)

    # 데이터셋 셔플
    train_data = list(zip(train_images, train_labels))
    np.random.shuffle(train_data)

    test_data = list(zip(test_images, test_labels))
    np.random.shuffle(test_data)

    return train_data, test_data


def train_image_dqn(train_data, train_labels, agent: ImageDQNAgent, batch_size=32, episodes=5000, target_tolerance=0.05, save_dir="./models", image_save_dir="./images"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)  # 이미지를 저장할 디렉토리 생성

    for e in range(episodes):
        total_reward = 0
        steps = 0
        done = False

        # 랜덤하게 하나의 이미지와 그에 대응하는 라벨을 선택
        idx = random.randint(0, len(train_data) - 1)
        state = train_data[idx]  # 이미지
        target_action = train_labels[idx]  # 해당 이미지의 좌표 (라벨)

        last_action = None  # 마지막 행동 저장 변수

        while not done:
            steps += 1

            # 에이전트의 행동 예측
            action = agent.act(state)

            # 예측된 행동과 실제 라벨 간 오차 계산 및 보상 부여
            reward = -np.linalg.norm(action - target_action)
            next_state = state  # 정적 이미지이므로 상태 변화가 없음

            # 에피소드 종료 조건: 오차가 target_tolerance 이하이거나 시도 횟수 255회 초과
            if np.linalg.norm(action - target_action) < target_tolerance or steps > 255:
                done = True

            # 에이전트 기억 업데이트
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            last_action = action  # 마지막 행동 저장

            if done:
                break

        # 리플레이 수행 (에이전트 학습)
        agent.replay(batch_size)

        # 마지막으로 계산된 중심과 현재 위치 간 오차 계산
        final_error = np.linalg.norm(last_action - target_action)
        print(f"Episode {e + 1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f} - Final Error: {final_error:.4f}")

        # 결과 이미지를 저장
        save_result_image(state, last_action, target_action, e + 1, image_save_dir)

    # 학습 완료 후 모델 저장
    final_model_path = os.path.join(save_dir, 'final_model.h5')
    agent.model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")


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


if __name__ == '__main__':
    print(device_lib.list_local_devices())

    # 학습 및 테스트 데이터 경로 설정
    train_csv = './data/train_labels.csv'
    train_folder = './data/train/balls'
    test_csv = './data/test_labels.csv'
    test_folder = './data/test/balls'

    # 데이터셋 로드
    train_data, test_data = load_dataset(train_csv, train_folder, test_csv, test_folder)

    # 모델 학습을 위해 데이터를 분리
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)

    print(f"loaded: {len(train_data)} / {len(test_images)}")

    agent = ImageDQNAgent()

    print("DQN Start!")
    train_image_dqn(train_images, train_labels, agent)

