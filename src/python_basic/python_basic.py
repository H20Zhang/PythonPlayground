import torch

class Pair:
    # __slots__ = ['x', 'y', "_zx"]  # variables inside slots are stored in fixed size array which saves a memory compared to
    # store variables inside dictionary. Though, it is a helpful features, you should resist to use it as many parts of
    # Python rely on the normal dictionary-based implementation.

    def __init__(self, x: int, y: int, z: int):
        self.x = x
        self.y = y
        self.zx = z

    #  Properties can be useful for computed attributes.
    @property
    def volumne(self):
        return self.x * self.y * self.z

    #  Use getter and setter if it is only necessary.
    #  Getter functions
    @property
    def zx(self):
        return self._zx

    # Setter functions
    @zx.setter
    def zx(self, value):
        self._zx = value

    def __repr__(self):
        """
        Hook into Python's string raw string outputting functionality.
        :return: code representation of the instance in string
        """
        return f"Pair({self.x}, {self.y}, {self.zx})"

    def __str__(self):
        """
        Hook into Python's string outputting functionality.
        :return: string representation of the instance
        """
        return f"({self.x}, {self.y}, {self.zx})"

    def __format__(self, format_spec):
        """
        Hook into Python's string formatting functionality.
        :param format_spec: formatted specs
        :return: formatted string with respect to specs
        """
        return f"({self.x}, {self.y}, {self.zx})"

    def __enter__(self):
        """
        Hook into Python's context management protocols.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Hook into Python's context management protocols.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        pass

    def _do_some_internal_thing(self):
        """
        Add _ to functions that are internal.
        Note: Python does not have access control
        """
        pass

    def __do_something_without_inheritance(self):
        """
        Add __ to functions that are not be inherited.
        """


if __name__ == '__main__':
    pair1 = Pair(1, 2, 3)
    print(pair1)
    print(dir(pair1))
    print("{0!r}".format(pair1))  # add !r to indicate to produce raw string


