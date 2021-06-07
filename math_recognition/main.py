from math_recognition.layoutpass import do_layout_pass, Symbol
from math_recognition.image_parser import parse_image, build_model, load_weights
from math_recognition.transformpass import do_transform
from math_recognition.symbols import symbol_to_str

from os import path
import cv2
import numpy as np


# Преобразует данные от нейронной сети в формат для layoutpass
def image_parser_data_converter(data):
    output = []
    symbols_imgs, symbols, symbols_bounds = data
    for i in range(len(symbols)):
        output.append([symbols[i], symbols_bounds[i]])

    return output


# Функция для перевода layoutpass в список
def layout_pass_to_list(layout: Symbol):
    list = []
    while layout is not None:
        if layout.above is not None:
            list.extend('(')
            list.extend(layout_pass_to_list(layout.above))
            list.extend(')')

        list.append(symbol_to_str(layout.symbol_label))

        if layout.super is not None:
            list.extend('^(')
            list.extend(layout_pass_to_list(layout.super))
            list.extend(')')
        if layout.subsc is not None:
            list.extend('_(')
            list.extend(layout_pass_to_list(layout.subsc))
            list.extend(')')
        if layout.below is not None:
            list.extend('(')
            list.extend(layout_pass_to_list(layout.below))
            list.extend(')')

        layout = layout.next
    return list


def prepare_network():
    build_model()
    checkpoint_path = path.join(path.dirname(__file__), 'training/cp.ckpt')
    load_weights(checkpoint_path)


# API 1
def math_recognition_image(img):
    output = parse_image(img)
    output = image_parser_data_converter(output)

    root_bt = do_layout_pass(output)
    do_transform(root_bt)
    return root_bt


# API 2
def math_recognition_image_by_path(path):
    img = cv2.imread(path)
    return math_recognition_image(img)


# mouse callback function
def paint_draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, mouse1_hold, mouse2_hold
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse1_hold = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        mouse2_hold = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse1_hold:
            cv2.line(image, (current_former_x, current_former_y), (former_x, former_y), (0, 0, 0), 5)
        if mouse2_hold:
            cv2.line(image, (current_former_x, current_former_y), (former_x, former_y), (255, 255, 255), 35)
        current_former_x = former_x
        current_former_y = former_y
    elif event == cv2.EVENT_LBUTTONUP:
        mouse1_hold = False
    elif event == cv2.EVENT_RBUTTONUP:
        mouse2_hold = False

    return former_x, former_y


if __name__ == "__main__":
    mouse1_hold = False  # true if mouse1 is pressed
    mouse2_hold = False  # true if mouse2 is pressed
    current_former_x = 0
    current_former_y = 0

    prepare_network()

    image = 255 * np.ones((500, 800, 3), dtype=np.uint8)
    cv2.namedWindow('Math-r Test')
    cv2.setMouseCallback('Math-r Test', paint_draw)
    while True:
        cv2.imshow('Math-r Test', image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Escape KEY
            break
        elif k == ord('c'):
            image[:] = 255
        elif k == ord('r'):
            # cv2.imwrite("tmp_expression.jpg", image)

            result = parse_image(image)
            result = image_parser_data_converter(result)

            x = do_layout_pass(result)
            do_transform(x)
            str_simple = ''.join(layout_pass_to_list(x))
            print(str_simple)

    cv2.destroyAllWindows()
