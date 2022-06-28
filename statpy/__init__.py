class dataset(list):
    from typing import Union

    class __sample_class:
        class sample:
            from typing import Union

            def __init__(self,  val: bool, obj: 'dataset'):
                self.val = val
                self.obj = obj

            def __call__(self, percent: Union[all, float] = 0.25):
                if percent == all:
                    percent = 1
                if percent > 1:
                    percent /= len(self.obj)
                if percent > 1:
                    raise IndexError(f"Your dataset only has {len(self.obj)} items and therefore"
                                     f" does not have {percent * len(self.obj)} items to sample")
                sample_length = round(percent * len(self.obj))
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

        def __init__(self, obj):
            self.obj = obj

        def __get__(self, instance, owner):
            return self.sample(instance._dataset__is_sample, self.obj)

        def __set__(self, instance, value: bool):
            if type(value) == self.sample:
                instance._dataset__is_sample = value.val
                return
            if value not in [True, False]:
                raise TypeError("sample must be either True or False")
            instance._dataset__is_sample = value

    def __new__(cls, __iterable=None):
        self = super().__new__(cls)
        cls.sample = cls.__sample_class(self)
        cls.__is_sample = False
        return self

    def __init__(self, __iterable=None):
        from random import randint
        self.randint = randint
        if __iterable is None:
            super().__init__()
            return
        if __iterable[0] == "SAMPLE":
            __iterable = __iterable.copy()
            __iterable.pop(0)
            self.__is_sample = True
        super().__init__(__iterable)

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

    @classmethod
    def __open_csv(cls, filename, header, delimiter=','):
        index = []
        data = dataset()
        with open(filename, 'r') as file:
            for line in file:
                line = line.split(delimiter)
                if not index:
                    if header[0] is all:
                        for i in range(len(line)):
                            data.append([])
                            index.append(i)
                        continue
                    for val in header:
                        data.append([])
                        index.append(line.index(val))
                else:
                    good = True
                    for val in index:
                        if line[val] == '':
                            good = False
                    if not good:
                        continue
                    for i, val in enumerate(index):
                        try:
                            line = line[val].strip()
                            if len(line.split('.')) > 1:
                                num = float(line)
                            else:
                                num = int(line)
                            data[i].append(num)
                        except ValueError:
                            pass
            file.close()
        return dataset(data)

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

    def copy(self) -> 'dataset':
        return dataset([x for x in self])


a = dataset([1, 2, 3, 4, 5])
a.sample = True
print(a)

import statpy