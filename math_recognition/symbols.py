from enum import Enum


class Symbols(Enum):
    SYMBOL_0 = 1
    SYMBOL_1 = 2
    SYMBOL_2 = 3
    SYMBOL_3 = 4
    SYMBOL_4 = 5
    SYMBOL_5 = 6
    SYMBOL_6 = 7
    SYMBOL_7 = 8
    SYMBOL_8 = 9
    SYMBOL_9 = 10
    SYMBOL_LBRACKET = 11
    SYMBOL_RBRACKET = 12
    SYMBOL_PLUS = 13
    SYMBOL_MINUS = 14
    SYMBOL_DOT = 15
    SYMBOL_X = 16
    SYMBOL_DECIMAL_DOT = 100
    SYMBOL_FRACTION = 101
    SYMBOL_MUL = 102
    SYMBOL_UNDERSCORE = 103
    SYMBOL_POWER = 104
    SYMBOL_EQUAL = 105


class SymbolType(Enum):
    DIGIT = 1
    OPERATOR = 2
    BRACKET = 3
    DECIMAL_DOT = 4
    COMPARISON = 5
    VARIABLE = 6
    OTHER = 100


def symbol_type(s):
    if s == Symbols.SYMBOL_0 or \
            s == Symbols.SYMBOL_1 or \
            s == Symbols.SYMBOL_2 or \
            s == Symbols.SYMBOL_3 or \
            s == Symbols.SYMBOL_4 or \
            s == Symbols.SYMBOL_5 or \
            s == Symbols.SYMBOL_6 or \
            s == Symbols.SYMBOL_7 or \
            s == Symbols.SYMBOL_8 or \
            s == Symbols.SYMBOL_9:
        return SymbolType.DIGIT
    elif s == Symbols.SYMBOL_PLUS or \
            s == Symbols.SYMBOL_MINUS or \
            s == Symbols.SYMBOL_FRACTION or \
            s == Symbols.SYMBOL_MUL or \
            s == Symbols.SYMBOL_POWER or \
            s == Symbols.SYMBOL_UNDERSCORE:
        return SymbolType.OPERATOR
    elif s == Symbols.SYMBOL_LBRACKET or \
            s == Symbols.SYMBOL_RBRACKET:
        return SymbolType.BRACKET
    elif s == Symbols.SYMBOL_DECIMAL_DOT:
        return SymbolType.DECIMAL_DOT
    elif s == Symbols.SYMBOL_EQUAL:
        return SymbolType.COMPARISON
    elif s == Symbols.SYMBOL_X:
        return SymbolType.VARIABLE
    else:
        return SymbolType.OTHER


def symbol_to_str(s):
    if s == Symbols.SYMBOL_0:
        return '0'
    elif s == Symbols.SYMBOL_1:
        return '1'
    elif s == Symbols.SYMBOL_2:
        return '2'
    elif s == Symbols.SYMBOL_3:
        return '3'
    elif s == Symbols.SYMBOL_4:
        return '4'
    elif s == Symbols.SYMBOL_5:
        return '5'
    elif s == Symbols.SYMBOL_6:
        return '6'
    elif s == Symbols.SYMBOL_7:
        return '7'
    elif s == Symbols.SYMBOL_8:
        return '8'
    elif s == Symbols.SYMBOL_9:
        return '9'
    elif s == Symbols.SYMBOL_PLUS:
        return '+'
    elif s == Symbols.SYMBOL_MINUS:
        return '-'
    elif s == Symbols.SYMBOL_MUL:
        return '*'
    elif s == Symbols.SYMBOL_DOT:
        return 'dot'
    elif s == Symbols.SYMBOL_LBRACKET:
        return '('
    elif s == Symbols.SYMBOL_RBRACKET:
        return ')'
    elif s == Symbols.SYMBOL_DECIMAL_DOT:
        return '.'
    elif s == Symbols.SYMBOL_FRACTION:
        return '/'
    elif s == Symbols.SYMBOL_POWER:
        return '^'
    elif s == Symbols.SYMBOL_UNDERSCORE:
        return '_'
    elif s == Symbols.SYMBOL_EQUAL:
        return '='
    elif s == Symbols.SYMBOL_X:
        return 'x'
    else:
        return ''


