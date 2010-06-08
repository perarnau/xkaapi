#!/usr/bin/env python


class Sample:

    def __init__(self):
        self._sample = []
        self._title = None
        self._average = None
        self._variance = None
        self._std_deviation = None
        return


    def load_file(filename):
        samples = None
        titles = []
        f = open(filename, "r")
        if f == None:
            return None

        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            cols = line.split()

            if len(titles) == 0 and str.isdigit(cols[0][0]) == False:
                titles = cols
                continue

            if samples == None:
                samples = []
                for c in cols:
                    samples.append(Sample())
                j = 0
                for t in titles:
                    samples[j]._title = t
                    j += 1

            i = 0
            for c in cols:
                try:
                    samples[i]._sample.append(int(c))
                except(ValueError):
                    samples[i]._sample.append(float(c))
                i += 1

        return samples

    load_file = staticmethod(load_file)


    def dump(self):
        print(str(len(self._sample)))
        for s in self._sample:
            print(str(s))


    def average(self):
        if self._average == None:
            size = len(self._sample)
            self._average = sum(self._sample) / size
        return self._average


    def variance(self):
        if self._variance == None:
            n = len(self._sample)
            a = self.average()**2
            s = sum([ x**2 for x in self._sample ])
            self._variance = s / n - a
        return self._variance


    def std_deviation(self):
        # ecart type
        import math
        if self._std_deviation == None:
            variance = self.variance()
            self._std_deviation = math.sqrt(variance)
        return self._std_deviation


    def deviation_coeff(self):
        return self.std_deviation() / self.average()


    def title(self):
        if self._title == None:
            return ''
        return self._title


def main(av):
    if (len(av[1]) < 2):
        return
    samples = Sample.load_file(av[1])
    if samples == None:
        return
    for s in samples:
        print(s.title() + ' ' + str(s.average()) + ' ' + str(s.std_deviation()) + ' ' + str(s.deviation_coeff()))
    return


if __name__ == '__main__':
    import sys
    main(sys.argv)
