

class Symbol:
    """
    Класс символа с координатами и распознанным значением.
    Координаты (x, y, w, h) можно трактовать как локальные
    относительно выреза ячейки, или как глобальные — 
    по вашему выбору.
    """

    def __init__(self, x, y, w, h, value=None, symbol_image=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.value = value
        self.symbol_image = symbol_image
    

    def __repr__(self):
        return f"Symbol(x={self.x}, y={self.y}, w={self.w}, h={self.h}, value={self.value})"