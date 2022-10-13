class __meta(type):
    def __getitem__(cls, *items) -> 'dataset':
        if cls == Vector:
            return cls(*items[0])
        return cls(items[0])


class dataset(list, metaclass=__meta):
    from typing import Union
    from random import randint
    __slots = ['__is_sample']

    class __sample:
        from typing import Union

        def __init__(self,  val: bool, obj: 'dataset'):
            self.val = val
            self.obj = obj

        def __call__(self, percent: Union[all, float] = 0.5):
            if percent == all:
                percent = 1
            if percent > 1:
                percent /= len(self.obj)
            if percent > 1:
                raise IndexError(f"Your dataset only has {len(self.obj)} items and therefore"
                                 f" does not have {percent * len(self.obj)} items to sample")
            sample_length = round(percent * len(self.obj))
            if percent and not sample_length:
                sample_length = 1
            if 1-percent and not len(self.obj)-sample_length:
                sample_length -= 1
            data = self.obj.copy()
            lst = ['SAMPLE']
            for _ in range(sample_length):
                i = self.obj.randint(0, len(data) - 1)
                lst.append(data.pop(i))
            return dataset(lst)

        def __repr__(self):
            return self.val

        def __str__(self):
            return str(self.val)

        def __bool__(self):
            return self.val

    def __init__(self, __iterable=None, labels=None):
        self.labels = labels
        self.__is_sample = False
        if __iterable in [None, [], (), {}]:
            super().__init__()
            return
        __iterable = list(__iterable)
        if __iterable[0] == "SAMPLE":
            __iterable = __iterable.copy()
            __iterable.pop(0)
            self.__is_sample = True
        for i, item in enumerate(__iterable):
            if '__iter__' in dir(item):
                __iterable[i] = Vector(*item)
        super().__init__(__iterable)

    def fit_polynomial(self, degree=None, acceptable_uncertainty=0.0001, starting_degree=0, minimum_uncertainty=None):
        if not (len(self) > 0 and (isinstance(self[0], Vector)) or (isinstance(self[0], tuple) and len(self[0]) == 2)):
            return
        temp = regression(self)
        temp.fit_polynomial(degree, acceptable_uncertainty, starting_degree, minimum_uncertainty)
        return temp

    @property
    def sample(self):
        return self.__sample(self.__is_sample, self)

    @sample.setter
    def sample(self, value):
        if type(value) == self.__sample:
            self.__is_sample = value.val
            return
        if value not in [True, False]:
            raise TypeError("sample must be either True or False")
        self.__is_sample = value

    def isample(self, percent: Union[all, float] = 0.25):
        """This is an in place sample"""
        if percent == all:
            percent = 1
        if percent > 1:
            percent /= len(self)
        if percent > 1:
            raise IndexError(f"Your dataset only has {len(self)} items and therefore"
                             f" does not have {percent * len(self)} items to sample")
        sample_length = round(percent * len(self))
        data = self.copy()
        lst = ['SAMPLE']
        for _ in range(sample_length):
            i = self.randint(0, len(data) - 1)
            lst.append(data.pop(i))
        lst = dataset(lst)
        while len(self) > 0:
            self.pop()
        self.extend(lst)

    @classmethod
    def open(cls, filename, *headers: Union[str, all]) -> 'dataset':
        file_type = filename.split('.')[-1]
        if file_type == 'csv':
            return cls.__open_csv(filename, headers)
        if file_type == 'tsv':
            return cls.__open_csv(filename, headers, delimiter='\t')
        return cls.__open_txt(filename)

    @classmethod
    def __open_txt(cls, filename):
        lst = []
        with open(filename, 'r') as file:
            for line in file:
                try:
                    line = line.strip()
                    if len(line.split('.')) > 1:
                        num = float(line)
                    else:
                        num = int(line)
                    lst.append(num)
                except ValueError:
                    pass
        return dataset(lst)

    def extend(self, __iterable):
        if isinstance(__iterable, dataset):
            super().extend(dataset([x for x in __iterable]))
        super().extend(__iterable)

    @classmethod
    def __open_csv(cls, filename, header, delimiter=','):
        index = []
        data = []
        labels = []
        do_vector = False
        with open(filename, 'r') as file:
            for line in file:
                line = line.split(delimiter)
                line = [x.strip() for x in line]
                if not index:
                    if header[0] is all:
                        for i in range(len(line)):
                            index.append(i)
                        if len(index) > 1:
                            do_vector = True
                        continue
                    for val in header:
                        if val not in line:
                            if val not in labels:
                                labels.append(val)
                            continue
                        index.append(line.index(val))
                        while val in labels:
                            labels.remove(val)
                    if len(index) > 1:
                        do_vector = True
                else:
                    good = True
                    for val in index:
                        if val >= len(line) or line[val] == '':
                            good = False
                    if not good:
                        continue
                    if do_vector:
                        temp = []
                        for val in index:
                            try:
                                current_line = line[val].strip()
                                if len(current_line.split('.')) > 1:
                                    num = float(current_line)
                                else:
                                    num = int(current_line)
                                temp.append(num)
                            except ValueError:
                                break
                        else:
                            data.append(Vector(*temp))
                        continue
                    for i, val in enumerate(index):
                        try:
                            current_line = line[val].strip()
                            if len(current_line.split('.')) > 1:
                                num = float(current_line)
                            else:
                                num = int(current_line)
                            data.append(num)
                        except ValueError:
                            pass
            file.close()
        data = dataset(data, labels)
        return data

    @property
    def sets(self):
        sets = []
        for i in range(len(self[0])):
            sets.append(dataset([vec[i] for vec in self]))
        return sets

    @property
    def mean(self) -> float:
        return sum(self) / len(self)

    @property
    def median(self) -> tuple:
        temp = self
        temp.sort()
        index = len(temp)
        if index % 2 == 0:
            return temp[index // 2],
        return temp[index // 2], temp[index // 2 + 1]

    @property
    def mode(self) -> tuple:
        modes = []
        lst = sorted(set(self), key=self.count, reverse=True)
        for i in range(len(lst) - 1):
            modes.append(lst[i])
            if self.count(lst[i]) != self.count(lst[i + 1]):
                break
        return tuple(modes)

    @property
    def range(self) -> float:
        return max(self) - min(self)

    @property
    def span(self):
        class span(list):
            @property
            def lower(self):
                return self[0]

            @property
            def upper(self):
                return self[1]

        return span([min(self), max(self)])

    @property
    def standard_deviation(self) -> float:
        return self.variance ** 0.5

    @property
    def stdev(self) -> float:
        return self.standard_deviation

    @property
    def variance(self) -> float:
        mean = self.mean
        total = 0
        for x in self:
            total += abs(x - mean) ** 2
        if self.__is_sample:
            return total / (len(self) - 1)
        return total / len(self)

    @property
    def uncertainty(self) -> float:
        return self.standard_deviation / (len(self) ** 0.5)

    def append(self, __object, __set_label=None) -> None:
        if isinstance(__object, str):
            labels = list(self.labels)
            labels.append(__object)
            self.labels = tuple(labels)
            for i, vec in enumerate(self):
                if not isinstance(vec, Vector):
                    for j, val in enumerate(self):
                        self[j] = Vector(val)
                    vec = Vector(vec)
                vec = list(vec.vector)
                vec.append(None)
                self[i].vector = tuple(vec)
            return
        if __set_label is not None:
            if __set_label not in self.labels:
                self.append(__set_label)
            index = self.labels.index(__set_label)
            for i in range(len(self)):
                if not isinstance(self[i], Vector):
                    for j, val in enumerate(self):
                        self[j] = Vector(val)
                if self[i][index] is None:
                    vec = list(self[i])
                    vec[index] = __object
                    self[i].vector = tuple(vec)
                    break
            else:
                temp = [None] * len(self[0])
                temp[index] = __object
                self.append(Vector(*temp))
            return
        super().append(__object)

    def copy(self) -> 'dataset':
        return dataset([x for x in self])

    def deepcopy(self):
        from copy import deepcopy
        return deepcopy(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            if len(item.split('.')) == 1:
                if item not in self.labels:
                    self.append(item)
                return self.sets[self.labels.index(item)]
            return dataset[zip(*[self[x] for x in item.split('.')])]
        return super().__getitem__(item)

    def __setattr__(self, key, value):
        if key in ['_dataset' + x for x in dataset.__slots] or key in ['sample', 'labels']:
            super().__setattr__(key, value)

    def __str__(self):
        if len(self) > 0 and isinstance(self[0], Vector):
            start = '['
            out = ', '
            out = out.join([str(x) for x in self])
            out += ']'
            return start + out
        return super().__str__()


class Vector(metaclass=__meta):
    def __init__(self, *args: float):
        from math import atan
        self.__atan = atan
        self.vector = args

    @property
    def magnitude(self):
        return sum([x**2 for x in self.vector]) ** 0.5

    @property
    def coordinates(self):
        return tuple(self.vector)

    @property
    def direction(self):
        x = self.vector[0]
        y = self.vector[1]
        if x == 0:
            if y > 0:
                direction = 90
            elif y < 0:
                direction = 270
            else:
                direction = 0
        else:
            direction = deg(self.__atan(y / x))
            if x < 0:
                direction += 180
        return direction

    @property
    def x(self):
        return self.vector[0]

    @x.setter
    def x(self, value):
        temp = list(self.vector)
        temp[0] = value
        self.vector = tuple(temp)

    @property
    def y(self):
        return self.vector[1]

    @y.setter
    def y(self, value):
        temp = list(self.vector)
        temp[1] = value
        self.vector = tuple(temp)

    @property
    def normalized(self):
        magnitude = self.magnitude
        return Vector(*[x / magnitude for x in self.vector])

    def __truediv__(self, other):
        if type(other) in [int, float]:
            return Vector(*[x / other for x in self.vector])

    def __mul__(self, other):
        if type(other) in [int, float]:
            return Vector(*[x * other for x in self.vector])
        if type(other) in [Vector]:
            return sum([v1*v2 for v1, v2 in zip(self.vector, other.vector)])

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return Vector(*[-x for x in self.vector])

    def __add__(self, other):
        return Vector(*[v1 + v2 for v1, v2 in zip(self.vector, other.vector)])

    def __sub__(self, other):
        return Vector(*[v1 - v2 for v1, v2 in zip(self.vector, other.vector)])

    def __getitem__(self, index):
        return self.vector[index]

    def __setitem__(self, key, value):
        vec = list(self)
        vec[key] = value
        self.vector = tuple(vec)

    def __str__(self):
        string = ', '.join([str(x) for x in self.vector])
        return f"<{string}>"

    def __len__(self):
        return len(self.vector)

    def __round__(self, n=None):
        return Vector(*[round(vec, n) for vec in self.vector])


def deg(radians):
    from math import pi
    return radians / pi * 180


def rad(degrees):
    from math import pi
    return degrees / 180 * pi


class regression:
    def __init__(self, points=None):
        self.points = []
        self.formula = None
        self.function = lambda x: x
        self.stdev = None
        self.type = None
        self.r_squared = None
        self.sorted_equations = None
        self.current = 0
        self.coefs = None
        if points is not None:
            for pt in points:
                self.append(pt)

    def fit_linear(self):
        x_bar = sum([vec.x for vec in self.points]) / len(self)
        y_bar = sum([vec.y for vec in self.points]) / len(self)
        slope = sum([(vec.x - x_bar) * (vec.y - y_bar) for vec in self.points]) / \
                sum([(vec.x - x_bar) ** 2 for vec in self.points])
        y_intercept = y_bar - slope * x_bar
        x_sum = sum([vec.x for vec in self.points])
        y_sum = sum([vec.y for vec in self.points])
        self.r_squared = ((len(self) * sum([vec.x * vec.y for vec in self.points]) - x_sum * y_sum) /
                          ((len(self) * sum([vec.x ** 2 for vec in self.points]) - x_sum ** 2) *
                           (len(self) * sum([vec.y ** 2 for vec in self.points]) - y_sum ** 2)) ** 0.5) ** 2
        self.function = lambda x: slope * x + y_intercept
        self.stdev = (1 / len(self) * sum([(vec.y - self.function(vec.x)) ** 2 for vec in self.points])) ** 0.5
        sign = '+'
        if y_intercept < 0:
            sign = '-'
        self.formula = f"y={slope}x{sign}{abs(y_intercept)}"
        self.type = "linear"
        return self.function

    def fit_quadratic(self):
        import numpy as np
        x = sum([vec.x for vec in self.points])
        y = sum([vec.y for vec in self.points])
        xy = sum([vec.x * vec.y for vec in self.points])
        x2 = sum([vec.x ** 2 for vec in self.points])
        x3 = sum([vec.x ** 3 for vec in self.points])
        x4 = sum([vec.x ** 4 for vec in self.points])
        x2y = sum([vec.x ** 2 * vec.y for vec in self.points])
        n = len(self)
        a1 = np.array([x2y, xy, y])
        b1 = np.matrix([[x4, x3, x2],
                        [x3, x2, x],
                        [x2, x, n]])
        sol = np.dot(a1, b1.I)
        a, b, c = [sol.item(i) for i in range(3)]
        self.function = lambda x: a * x ** 2 + b * x + c
        self.stdev = (1 / len(self) * sum([(vec.y - self.function(vec.x)) ** 2 for vec in self.points])) ** 0.5
        self.formula = f"y={a}x^2+{b}x+{c}"
        self.type = "quadratic"
        return self.function

    def fit_polynomial(self, degree=None, acceptable_uncertainty=0.0001, starting_degree=0, minimum_uncertainty=None):
        import numpy as np
        current_degree = starting_degree
        if degree is not None:
            current_degree = int(degree)
        while True:
            X = np.matrix([[vec.x ** d for d in range(current_degree + 1)] for vec in self.points])
            Xt = X.transpose()
            Y = np.array([vec.y for vec in self.points])
            B = ((Xt * X).I * Xt).dot(Y)
            coefficients = [B.item(x) for x in range(current_degree + 1)]
            self.coefs = coefficients.copy()
            self.coefs.reverse()
            func = lambda x: sum([coefficients[i] * x ** i for i in range(len(coefficients))])
            stdev = (1 / len(self) * sum([(vec.y - func(vec.x)) ** 2 for vec in self.points])) ** 0.5
            if self.stdev is not None and (self.stdev <= stdev or self.stdev < acceptable_uncertainty) and \
                    (minimum_uncertainty is None or self.stdev <= minimum_uncertainty):
                break
            self.function = func
            self.stdev = stdev
            self.formula = ''.join([f"{coefficients[i]}x^{i} + " for i in range(len(coefficients) - 1, -1, -1)]) + \
                           "\b\b\b\b\b\b"
            current_degree += 1
            if degree is not None:
                break
        self.type = "polynomial"
        return self.function

    def fit_exp(self, use_raw=False):
        import numpy as np
        x = [vec.x for vec in self.points]
        y = [vec.y for vec in self.points]
        c = 0
        if not use_raw:
            c = min(y) - 0.001
            y = [i-c for i in y]
        elif min(y) < 0:
            return False
        A = np.vstack([x, np.ones(len(x))]).T
        b, log_alpha = np.linalg.lstsq(A, np.log(y), rcond=None)[0]
        a = np.exp(log_alpha)
        func = lambda x: a * np.exp(b * x) + c
        self.function = func
        self.stdev = (1 / len(self) * sum([(vec.y - func(vec.x)) ** 2 for vec in self.points])) ** 0.5
        self.type = "exponential"
        if not use_raw:
            self.formula = f"{a}e^({b}x) + {c}"
        else:
            self.formula = f"{a}e^({b}x)"
            return True
        return self.function

    def fit(self):
        results = []
        #linear
        self.fit_linear()
        results.append({"stdev": self.stdev, "function": self.function,
                        "formula": self.formula, "type": self.type, "r_squared": self.r_squared})
        #quadratic
        self.fit_quadratic()
        results.append({"stdev": self.stdev, "function": self.function,
                        "formula": self.formula, "type": self.type, "r_squared": None})
        #other polynomials (up to power of 50)
        for i in range(3, 50):
            self.fit_polynomial(i)
            results.append({"stdev": self.stdev, "function": self.function,
                            "formula": self.formula, "type": self.type, "r_squared": None})
        #exponential
        self.fit_exp()
        results.append({"stdev": self.stdev, "function": self.function,
                        "formula": self.formula, "type": self.type, "r_squared": None})
        if self.fit_exp(use_raw=True):
            results.append({"stdev": self.stdev, "function": self.function,
                            "formula": self.formula, "type": self.type, "r_squared": None})

        #sort results and find best function
        results.sort(key=lambda x: x["stdev"])
        self.sorted_equations = results
        self.function = results[0]["function"]
        self.stdev = results[0]["stdev"]
        self.formula = results[0]["formula"]
        self.r_squared = results[0]["r_squared"]
        self.type = results[0]["type"]

    def next(self):
        if self.sorted_equations is None:
            return
        old = self.current
        self.current += 1
        if self.current > len(self.sorted_equations) - 1:
            self.current = 0
        for _ in range(len(self.sorted_equations)):
            if self.sorted_equations[old]["type"] != self.sorted_equations[self.current]["type"]:
                break
            self.current += 1
            if self.current > len(self.sorted_equations) - 1:
                self.current = 0
        results = self.sorted_equations[self.current]
        self.function = results["function"]
        self.stdev = results["stdev"]
        self.formula = results["formula"]
        self.r_squared = results["r_squared"]
        self.type = results["type"]

    def append(self, vec):
        if type(vec) == list:
            for v in vec:
                self.append(v)
            return
        if type(vec) == tuple and len(vec) == 2:
            vec = Vector(vec[0], vec[1])
        if type(vec) != Vector:
            raise TypeError \
                (f"Failed to append {vec} please make sure that {vec} is "
                 f"either a Vector or a tuple of length 2. {vec} is currently a {type(vec)}")
        self.points.append(vec)

    def remove(self, vec):
        self.points.remove(vec)

    def pop(self, index=0):
        return self.points.pop(index)

    def set_equation(self, index):
        if self.sorted_equations is None:
            raise SystemError("There are no equations to choose from. The .fit() function is what creates the list "
                              "of equations")
        results = self.sorted_equations[index]
        self.function = results["function"]
        self.stdev = results["stdev"]
        self.formula = results["formula"]
        self.r_squared = results["r_squared"]
        self.type = results["type"]

    @property
    def uncertainty(self):
        if self.stdev is None:
            return None
        return self.stdev / (len(self)) ** 0.5

    @property
    def statistics(self):
        string = f"Standard Deviation: {self.stdev}"
        string += f"\nUncertainty: {self.uncertainty}"
        if self.r_squared is not None:
            string += f"\nr^2 Value: {self.r_squared}"
        if self.type is not None:
            string += f"\nEquation Class: {self.type}"
        return string

    def __len__(self):
        return len(self.points)

    def __str__(self):
        return self.formula

    def __call__(self, x):
        return self.function(x)
