__all__ = ["Symbol", "do_layout_pass"]

from enum import Enum
from collections import namedtuple
from math_recognition.symbols import Symbols
import math


Bounds = namedtuple('Bounds', ['left', 'top', 'right', 'bottom'])
Regions = namedtuple('Regions', ['next', 'super', 'subsc', 'above', 'below'])


class SymbolClass(Enum):
    NON_SCRIPTED = 1
    PLAIN_CENTERED = 2
    PLAIN_DESCENDER = 3
    PLAIN_ASCENDER = 4
    OPEN_BRACKET = 5


class Region(Enum):
    NEXT = 1
    SUPER = 2
    SUBSC = 3
    ABOVE = 4
    BELOW = 5


# Пока для классификации буду использовать словарь, потом может что-то другое придумаю
classes_dictionary = {
    Symbols.SYMBOL_X: SymbolClass.PLAIN_CENTERED,
    Symbols.SYMBOL_PLUS: SymbolClass.NON_SCRIPTED,
    Symbols.SYMBOL_MINUS: SymbolClass.NON_SCRIPTED,
    Symbols.SYMBOL_0: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_1: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_2: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_3: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_4: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_5: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_6: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_7: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_8: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_9: SymbolClass.PLAIN_ASCENDER,
    Symbols.SYMBOL_LBRACKET: SymbolClass.OPEN_BRACKET,
    Symbols.SYMBOL_RBRACKET: SymbolClass.PLAIN_CENTERED,
    Symbols.SYMBOL_DOT: SymbolClass.NON_SCRIPTED,
    Symbols.SYMBOL_EQUAL: SymbolClass.NON_SCRIPTED,
}

"""
Можно заменить потом classes_dictionary на это:

classes_dictionary = {
    'digit': SymbolClass.PLAIN_ASCENDER,
    'operator': SymbolClass.NON_SCRIPTED,
    # etc
}

def get_symbol_class(sybmol):
    if symbol.isdigit():
        return classes_dictionary[digit]
    elif ...
"""

# Глобальные переменные
centroid_ratio = 0.5
threshold_ratio = 0.2  # t <= c; t, c iz [0, 0.5]
###


class Symbol:
    """Класс, описывающий один символ из математического выражения"""
    __symbol_label = None
    __symbol_class = None  # SymbolClass.NON_SCRIPTED
    __next = None
    __super = None
    __subsc = None
    __above = None
    __below = None
    # И другие ещё ...
    # Везде будем использовать абсолютные координаты
    __bounds = None  # Bounds(0, 0, 0, 0)
    __centroid = None  # [x, y]
    __regions = None  # Regions(0, 0, 0, 0, 0)

    def __init__(self, label):
        self.__symbol_label = label

    @property
    def symbol_class(self):
        return self.__symbol_class

    @symbol_class.setter
    def symbol_class(self, sym_class):  # При изменении класса нужно также менять regions и т.д.?
        self.__symbol_class = sym_class

    @property
    def bounds(self):
        return self.__bounds

    @bounds.setter
    def bounds(self, b):
        self.__bounds = b

    @property
    def regions(self):
        return self.__regions

    @regions.setter
    def regions(self, b):
        self.__regions = b

    @property
    def centroid(self):
        return self.__centroid

    @centroid.setter
    def centroid(self, c):
        self.__centroid = c

    @property
    def symbol_label(self):
        return self.__symbol_label

    @symbol_label.setter
    def symbol_label(self, l):
        self.__symbol_label = l

    @property
    def next(self):
        return self.__next

    @next.setter
    def next(self, n):
        self.__next = n

    @property
    def super(self):
        return self.__super

    @super.setter
    def super(self, s):
        self.__super = s

    @property
    def subsc(self):
        return self.__subsc

    @subsc.setter
    def subsc(self, s):
        self.__subsc = s

    @property
    def below(self):
        return self.__below

    @below.setter
    def below(self, b):
        self.__below = b

    @property
    def above(self):
        return self.__above

    @above.setter
    def above(self, a):
        self.__above = a

    def about(self):
        print(self.__symbol_label)
        print(self.__symbol_class)
        print(self.__bounds)
        print(self.__centroid)
        print(self.__regions)

    def __eq__(self, other):  # Может, не надо так?
        if isinstance(other, Symbol):
            return id(self) == id(other)
        return False


# Функция для создания списка объектов Symbol из списка, полученного от модуля с нейронкой
def symbols_data_convertor(input_symbols_data):
    symbols_list = []
    for x in input_symbols_data:
        s = Symbol(x[0])
        s.bounds = Bounds(left=x[1][0], top=x[1][1], right=x[1][0] + x[1][2], bottom=x[1][1] + x[1][3])
        s.symbol_class = classes_dictionary[s.symbol_label]
        set_symbol_thresholds_and_centroid(s)

        symbols_list.append(s)

    return symbols_list


# Параметры могут отлючаться от табличных
# Нужно перепроверить..
def set_symbol_thresholds_and_centroid(symbol):
    H = symbol.bounds.bottom - symbol.bounds.top
    if symbol.symbol_class == SymbolClass.NON_SCRIPTED:
        symbol.centroid = [(symbol.bounds.right + symbol.bounds.left) / 2,
                           symbol.bounds.bottom - 1 / 2 * H]
        symbol.regions = Regions(next=symbol.bounds.right,
                                 super=-math.inf,  # Нужно не inf, а границы символа. Думаю, что будет лучше
                                 subsc=math.inf,  # ??? -inf?
                                 above=symbol.bounds.bottom - H / 2,
                                 below=symbol.bounds.bottom - H / 2)
    elif symbol.symbol_class == SymbolClass.PLAIN_ASCENDER:
        symbol.centroid = [(symbol.bounds.right + symbol.bounds.left) / 2,
                           symbol.bounds.bottom - centroid_ratio * H]
        symbol.regions = Regions(next=symbol.bounds.left,
                                 super=symbol.bounds.bottom - (H - threshold_ratio * H),
                                 subsc=symbol.bounds.bottom - threshold_ratio * H,
                                 above=symbol.bounds.bottom - (H - threshold_ratio * H),
                                 below=symbol.bounds.bottom - threshold_ratio * H)
    elif symbol.symbol_class == SymbolClass.PLAIN_DESCENDER:
        symbol.centroid = [(symbol.bounds.right + symbol.bounds.left) / 2,
                           symbol.bounds.bottom - (H - centroid_ratio * H)]
        symbol.regions = Regions(next=symbol.bounds.left,
                                 super=symbol.bounds.bottom - (H - threshold_ratio / 2 * H),
                                 subsc=symbol.bounds.bottom - (H / 2 + threshold_ratio / 2 * H),
                                 above=symbol.bounds.bottom - (H - threshold_ratio / 2 * H),
                                 below=symbol.bounds.bottom - (H / 2 + threshold_ratio / 2 * H))
    elif symbol.symbol_class == SymbolClass.PLAIN_CENTERED:
        symbol.centroid = [(symbol.bounds.right + symbol.bounds.left) / 2,
                           symbol.bounds.bottom - 1 / 2 * H]
        symbol.regions = Regions(next=symbol.bounds.left,
                                 super=symbol.bounds.bottom - (H - threshold_ratio * H),
                                 subsc=symbol.bounds.bottom - threshold_ratio * H,
                                 above=symbol.bounds.bottom - (H - threshold_ratio * H),
                                 below=symbol.bounds.bottom - threshold_ratio * H)
    elif symbol.symbol_class == SymbolClass.OPEN_BRACKET:
        symbol.centroid = [(symbol.bounds.right + symbol.bounds.left) / 2,
                           symbol.bounds.bottom - centroid_ratio * H]
        symbol.regions = Regions(next=symbol.bounds.left,
                                 super=-math.inf,  # Возможно, нужно написать super=inf, subsc=-inf, а next находить как-то по-другому
                                 subsc=math.inf,  # Например, если лежит справа, но не принадлежит никакому региону.
                                 above=symbol.bounds.top,
                                 below=symbol.bounds.bottom)


def sort_symbols_list(symbols):
    symbols.sort(key=lambda s: s.bounds.left)


# Пока так. Нужно продумать с примером 10.
def dominance(s1, s2):
    if s1.symbol_class == SymbolClass.NON_SCRIPTED and s1.bounds.left <= s2.centroid[0] < s1.bounds.right:
        if not ((s2.symbol_label == Symbols.SYMBOL_LBRACKET or s2.symbol_label == Symbols.SYMBOL_RBRACKET) and
                s2.bounds.top <= s1.centroid[1] < s2.bounds.bottom and
                s2.bounds.left <= s1.bounds.left):
            if not (s2.symbol_class == SymbolClass.NON_SCRIPTED and
                    s2.bounds.right - s2.bounds.left > s1.bounds.right - s1.bounds.left):
                return True
    return False


def find_start_symbol(symbols):
    L = symbols.copy()
    n = len(L)
    while n > 1:
        if dominance(L[n - 1], L[n - 2]):
            del L[n - 2]
        else:
            del L[n - 1]
        n = n - 1

    return L[0]


# Здесь будет <= (т.е. включительно), а в belong_region будет > (т.е. не включительно)
def is_adjacent(s1, s2):
    if s1.regions.super <= s2.centroid[1] <= s1.regions.subsc:
        return True

    return False


# Проверить границы (<= и <, >= и >)
def belong_region(s1, s2):
    x_cent = s2.centroid[0]
    y_cent = s2.centroid[1]

    if x_cent > s1.bounds.right and y_cent < s1.regions.super:
        return Region.SUPER
    elif x_cent > s1.bounds.right and y_cent > s1.regions.subsc:
        return Region.SUBSC
    elif s1.bounds.left <= x_cent <= s1.bounds.right and s1.regions.above > y_cent:
        if s1.symbol_class == SymbolClass.NON_SCRIPTED:  # Это временное решение, т.к. у нас нет диакритических знаков
            return Region.ABOVE
        else:
            return Region.SUPER  # Если символ находится НАД другим символом (не дробью), то он причисляется в SUPER
    elif s1.bounds.left <= x_cent <= s1.bounds.right and s1.regions.below < y_cent:
        return Region.BELOW

    # Дописать


# HOR | прочитать про различные HOR, которые начинаются с xmin или с xmax!!
def find_next_in_baseline(s_cur, symbols):
    for x in symbols:
        if x == s_cur or x.centroid[0] <= s_cur.regions.next:
            continue
        if is_adjacent(s_cur, x):
            return x

    return None


# Функция составляет новый список из символов, где два минуса друг над другом заменяются на "равно"
def __preprocessing_equal_sign(data):
    processed_data = []

    while len(data) > 0:
        s = data.pop()
        if s.symbol_label != Symbols.SYMBOL_MINUS:
            processed_data.append(s)
            continue

        second_line = None
        k = 0
        for s2 in data:
            if k > 1:
                break

            if s.bounds.left <= s2.centroid[0] <= s.bounds.right:
                k = k + 1
                if s2.symbol_label == Symbols.SYMBOL_MINUS:
                    second_line = s2

        if k == 1 and second_line is not None:
            # Если нет других символов над\под, кроме второй черты
            left = s.bounds.left if s.bounds.left < second_line.bounds.left else second_line.bounds.left
            right = s.bounds.right if s.bounds.right > second_line.bounds.right else second_line.bounds.right
            top = s.bounds.top if s.bounds.top < second_line.bounds.top else second_line.bounds.top
            bottom = s.bounds.bottom if s.bounds.bottom > second_line.bounds.bottom else second_line.bounds.bottom

            # Уже такой код есть => вынести в отдельную функцию
            s_new = Symbol(Symbols.SYMBOL_EQUAL)
            s_new.bounds = Bounds(left=left, top=top, right=right, bottom=bottom)
            s_new.symbol_class = classes_dictionary[s_new.symbol_label]
            set_symbol_thresholds_and_centroid(s_new)

            data.remove(second_line)
            processed_data.append(s_new)
        else:
            processed_data.append(s)

    return processed_data


def do_layout_pass(data):
    symbols = symbols_data_convertor(data)
    symbols = __preprocessing_equal_sign(symbols)

    return layout_pass(symbols)


def layout_pass(symbols):
    if symbols is None or len(symbols) == 0:
        return None

    sort_symbols_list(symbols)  # Нужна ли постоянная сортировка?
    s = find_start_symbol(symbols)
    next_s = find_next_in_baseline(s, symbols)

    regions_dict = dict()
    while len(symbols) > 0 and symbols[0] != next_s:
        if symbols[0] == s:
            symbols.pop(0)
            continue

        region = belong_region(s, symbols[0])
        if region not in regions_dict:
            regions_dict[region] = []

        regions_dict[region].append(symbols[0])
        symbols.pop(0)

    s.next = layout_pass(symbols)
    s.super = layout_pass(regions_dict.get(Region.SUPER, None))
    s.subsc = layout_pass(regions_dict.get(Region.SUBSC, None))
    s.above = layout_pass(regions_dict.get(Region.ABOVE, None))
    s.below = layout_pass(regions_dict.get(Region.BELOW, None))

    return s
