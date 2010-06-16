#!/usr/bin/env python


class Sample:

    def __init__(self):
        self._sample = []
        self._meta = {}
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
                    samples[j].set_meta('title', t)
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


    def set_meta(self, key, value):
        self._meta[key] = value
        return


    def get_meta(self, key):
        if self._meta.has_key(key) == False:
            return None
        return self._meta[key]


    def has_meta(self, meta):
        for k in meta:
            if self.get_meta(k) != meta[k]:
                return False
        return True


class SampleSet:

    def __init__(self):
        self._samples = []
        return


    def load_directory(self, pathname, on_sample):
        filenames = SampleSet.get_filenames(pathname)
        for filename in filenames:
            samples = Sample.load_file(filename)
            for sample in samples:
                on_sample(sample, filename)
                self._samples.append(sample)
        return


    def get_filenames(pathname):
        # pathname the directory path
        # assume flat directory
        import os
        res = []
        for dirname, foo, filenames in os.walk(pathname):
            filenames.sort()
            for filename in filenames:
                res.append(os.path.join(dirname, filename))
        return res

    get_filenames = staticmethod(get_filenames)


    def find_samples(self, meta):
        samples = []
        for s in self._samples:
            if s.has_meta(meta) == True:
                samples.append(s)
        return samples


# user code

def on_sample(sample, filename):
    nods = filename.split('/')[-1]
    toks = nods.split('-')
    sample.set_meta('algo', toks[0])
    sample.set_meta('impl', toks[1])
    sample.set_meta('seq_size', toks[3])
    sample.set_meta('cpu_count', toks[4])
    return

def cpu_count_to_size(n):
    counts = ('1 ', '2 ', '4 ', '8 ', '12', '16')
    #counts = ('8 ', '9 ', '10', '11', '12', '13', '14', '15', '16')
    return counts[int(n)]

def main(dirname):
    ss = SampleSet()
    ss.load_directory(dirname, on_sample)

    for size in [100000]:
        print('-- sequence_size: ' + str(size))
        for impl in ['kastl', 'stl']:
            print('---- impl: ' + impl)
            for algo in [\
                "transform",\
                    "for_each",\
                "find",\
                "find_if",\
                "find_first_of",\
                "accumulate",\
                "inner_product",\
                "count",\
                "count_if",\
                "copy",\
                "search",\
                "min_element",\
                "max_element",\
                "fill",\
                "generate",\
                "inner_product",\
                "replace",\
                "replace_if",\
                "equal",\
                "mismatch",\
                "search",\
                "adjacent_find",\
                "adjacent_difference"
                ]:
                samples = ss.find_samples\
                    ({'algo': algo, 'impl': impl, 'seq_size': str(size)})
                print('---- algo: ' + algo)
                for s in samples:
                    print(cpu_count_to_size(s.get_meta('cpu_count'))\
                              + ': ' + str(s.average()))
                print('')

#     seq_avg = None
#     for filename in filenames:
#         sample = find_time_sample(Sample.load_file(filename))
#         if sample == None:
#             continue
#         time_avg = sample.average()
#         if (seq_avg == None):
#             seq_avg = time_avg
#         cpu_count = int(get_last_name(filename))
#         speedup_avg = seq_avg / time_avg
#         # confident interval: 95% with 2 * time_dc
#         time_dc = sample.deviation_coeff() * 2
#         speedup_avg_low = speedup_avg + time_dc * speedup_avg
#         speedup_avg_high = speedup_avg - time_dc * speedup_avg
#         print(str(cpu_count) + ' ' + str(speedup_avg) + ' ' + str(speedup_avg_low) + ' ' + str(speedup_avg_high))
    return


if __name__ == '__main__':
    import sys
    dirname = '../session/this/'
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    main(dirname)
