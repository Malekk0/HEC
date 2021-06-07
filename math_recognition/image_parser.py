__all__ = ["parse_image", "build_model", "load_weights"]

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import cv2
import matplotlib.pyplot as plt
from math_recognition.symbols import Symbols

model = keras.Sequential()
symbol_image_size = 48

symbols_dictionary = {
    0: Symbols.SYMBOL_0,
    1: Symbols.SYMBOL_1,
    2: Symbols.SYMBOL_2,
    3: Symbols.SYMBOL_3,
    4: Symbols.SYMBOL_4,
    5: Symbols.SYMBOL_5,
    6: Symbols.SYMBOL_6,
    7: Symbols.SYMBOL_7,
    8: Symbols.SYMBOL_8,
    9: Symbols.SYMBOL_9,
    10: Symbols.SYMBOL_PLUS,
    11: Symbols.SYMBOL_MINUS,
    12: Symbols.SYMBOL_DOT,
    13: Symbols.SYMBOL_LBRACKET,
    14: Symbols.SYMBOL_RBRACKET,
    15: Symbols.SYMBOL_X,
}


def build_model():
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same', input_shape=(symbol_image_size, symbol_image_size, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])


def load_weights(path):
    model.load_weights(path)
    print("loaded.")


def scale_contour(cnt, scale):
    # M = cv2.moments(cnt)
    (x, y, w, h) = cv2.boundingRect(cnt)
    cx = x  # int(M['m10']/M['m00'])
    cy = y  # int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def parse_image(img):
    # Подгружаем картинку
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Достаём отдельные контуры из картинки
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE

    # Формируем массив символов из полученных ранее контуров; сжимаем каждую картинку
    # символа до (symbol_image_size на symbol_image_size)
    symbols = []
    symbols_bounds = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == 0:
            (x, y, w, h) = cv2.boundingRect(contour)

            # Отбрасываем символы, площадь рамки которых меньше порога
            if w * h <= 100:
                continue

            size_max = max(w, h)
            symbol_squared = 255 * np.ones((symbol_image_size, symbol_image_size))
            aspect = (symbol_image_size - 2.0) / size_max

            symbol = gray[y:y + h, x:x + w]
            symbol = cv2.bitwise_not(symbol)  # Так не дело

            # Вырезаем символ по маске, чтобы в прямоугольную область не попадали куски других символов
            mask = np.zeros(symbol.shape, np.uint8)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1, offset=(-x, -y))
            symbol = cv2.bitwise_and(symbol, symbol, mask=mask)
            symbol = cv2.bitwise_not(symbol)  # Так не дело

            # Сжимаем по бОльшей стороне
            symbol = cv2.resize(symbol, (int(np.ceil(w * aspect)), int(np.ceil(h * aspect))),
                                interpolation=cv2.INTER_LANCZOS4)  # Посмотреть другие interp
            symbol_size = symbol.shape

            if h > 300 or w > 300:  # Если символ слишком большой, то использовать не сжатое изображений, а сжатый контур
                test_img = 255 * np.ones((symbol_image_size, symbol_image_size))
                contour_scaled = scale_contour(contour, aspect)
                cv2.drawContours(test_img, [contour_scaled], 0, (0, 0, 0), 1, offset=(-x, -y))
                symbol = test_img

            # Создаём квадратный символ (symbol_image_size на symbol_image_size) и по центру помещаем исходный
            shiftW = int(symbol_image_size // 2 - symbol_size[1] // 2)
            shiftH = int(symbol_image_size // 2 - symbol_size[0] // 2)
            for i in range(symbol_size[0]):
                for j in range(symbol_size[1]):
                    symbol_squared[i + shiftH, j + shiftW] = symbol[i, j]

            # plt.imshow(symbol_squared, cmap=plt.cm.binary)
            # plt.show()

            symbols.append(symbol_squared)
            symbols_bounds.append([x, y, w, h])

    predicted_symbol_labels = []
    for id in range(len(symbols)):
        # Нормализируем
        symbols[id] = symbols[id] / 255.0
        symbols[id] = 1 - symbols[id]

        id_symbol_predicted = np.argmax(model.predict(np.array([symbols[id].reshape(symbol_image_size, symbol_image_size, 1)])))
        symbol_predicted = symbols_dictionary[id_symbol_predicted]

        predicted_symbol_labels.append(symbol_predicted)

    return symbols, predicted_symbol_labels, symbols_bounds


if __name__ == "__main__":
    build_model()
    checkpoint_path = "training/cp.ckpt"
    load_weights(checkpoint_path)

    result = parse_image(cv2.imread('expr_examples/expression4444.png'))

    fig = plt.figure(figsize=(8, 8))
    for i in range(len(result[0])):
        plt.subplot(int(np.ceil(len(result[0]) / 10)), 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(result[0][i], cmap=plt.cm.binary)
        plt.xlabel(str(result[1][i]))
    plt.show()
