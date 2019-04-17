"""
Representation of a single individual
Constructor takes dict with and genotype - 32bit floating point numbers with their names
Then fenotype is calculated
Only works on floating point numbers as real numbers are the most used

Possible additional features:
-allow user to set how may bytes does he want to use
"""

import struct


def float_to_bin(num):
    return format(struct.unpack("!I", struct.pack("!f", num))[0], "032b")


def bin_to_float(binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


class Individual:
    def __init__(self, genotype):
        self._genotype = genotype
        self._fenotype = {}
        self.calculate_fenotype()

    def calculate_fenotype(self):
        for key, value in self._genotype.items():
            self._fenotype[key] = float(bin_to_float(value))

    def fenotype(self):
        return self._fenotype

    def genotype(self):
        return self._genotype

    def __str__(self):
        return str(self._fenotype)
