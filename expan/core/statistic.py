from expan.core.jsonable import Jsonable

class Statistic(Jsonable):
    def __init__(self, name, value):
        self.name  = name
        self.value = value

    def __repr__(self):
        return 'Statistic: ' + str(self.name) + ' = ' + str(self.value)
