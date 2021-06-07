__all__ = ["do_transform"]

from math_recognition.layoutpass import Symbol
from math_recognition.symbols import Symbols


def do_transform(s: Symbol):
    transform(s)


def transform(s: Symbol):
    if s is None:
        return

    # Пока такое решение. Иногда точка попадает в below..
    if s.subsc is not None and s.subsc.symbol_label == Symbols.SYMBOL_DOT:
        s.subsc.symbol_label = Symbols.SYMBOL_DECIMAL_DOT
        s.subsc.next = s.next
        s.next = s.subsc
        s.subsc = None

    if s.symbol_label == Symbols.SYMBOL_DOT:  # Пока такое решение
        s.symbol_label = Symbols.SYMBOL_MUL

    if s.above is not None and \
            s.below is not None and \
            s.symbol_label == Symbols.SYMBOL_MINUS:
        s.symbol_label = Symbols.SYMBOL_FRACTION

    transform(s.next)
    transform(s.super)
    transform(s.subsc)
    transform(s.below)
    transform(s.above)


def __is_no_surroundings(s: Symbol):
    if s.above is None and s.below is None and \
            s.subsc is None and s.super is None and s.next is None:
        return True
    else:
        return False

