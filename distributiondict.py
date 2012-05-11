from collections import defaultdict

class DistributionDict:
    """
    Data Structure used for calculating the prior probabilities of sequences
    """

    def __init__(self):
        self._dict = defaultdict(int)
        self._key = '__INCR__'

    def add(self, *args):
        """add a sequence to the dictionary"""
        def _add(dictionary, *args):
            if len(args) == 0: return
            dictionary[self._key] += 1
            if not args[0] in dictionary:
                dictionary[args[0]] = defaultdict(int)
            if len(args) == 1:
                dictionary[args[0]][self._key] += 1
            _add(dictionary[args[0]], *args[1:])
        _add(self._dict, *args)

    def sum(self, *args):
        def _sum(total, dictionary, *args):
            if len(args) == 0:
                return total + dictionary[self._key]
            return _sum(total, dictionary[args[0]], *args[1:])
        return _sum(0.0, self._dict, *args[:-1])

    def probability(self, *args):
        """return the calculated probability for a sequence"""
        def _probability(dictionary, *args):
            if len(args) == 0:
                return dictionary[self._key] / self.sum(*args)
            return _probability(dictionary[args[0]], *args[1:])
        return _probability(self._dict, *args)


if __name__ == '__main__':
    d = DistributionDict()
    d.add(1)
    d.add(1,2)
    d.add(1,2,3)
    print d.probability(1,2)