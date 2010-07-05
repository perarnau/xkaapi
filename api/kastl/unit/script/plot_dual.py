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


def load_directory(pathname):
    import os
    res = []
    for dirname, foo, filenames in os.walk(pathname):
        filenames.sort()
        for filename in filenames:
            res.append(os.path.join(dirname, filename))
    return res


def get_last_name(filename):
    return filename.split('/')[-1]


def prepare_stats():
    s = {}
    s['seq_avg'] = None
    s['speedup_avg'] = None
    s['speedup_avg_low'] = None
    s['speedup_avg_high'] = None
    return s


def compute_stats(sample, stats):
    time_avg = sample.average()
    if (stats['seq_avg'] == None):
        stats['seq_avg'] = time_avg
    seq_avg = stats['seq_avg']
    speedup_avg = seq_avg / time_avg
    stats['speedup_avg'] = speedup_avg
    # interval de confiance: 95% avec 2 * time_dc
    time_dc = sample.deviation_coeff() * 2
    stats['speedup_avg_low'] = seq_avg / (time_avg + time_dc)
    stats['speedup_avg_high'] = seq_avg / (time_avg - time_dc)


def main(idkoiff_dirname, grimage_dirname):
    idkoiff_filenames = load_directory(idkoiff_dirname)
    grimage_filenames = load_directory(grimage_dirname)

    idkoiff_stats = prepare_stats()
    grimage_stats = prepare_stats()

    for i in range(0, len(idkoiff_filenames)):
        idkoiff_filename = idkoiff_filenames[i]
        grimage_filename = grimage_filenames[i]

        idkoiff_sample = Sample.load_file(idkoiff_filename)[0]
        grimage_sample = Sample.load_file(grimage_filename)[0]

        compute_stats(idkoiff_sample, idkoiff_stats)
        compute_stats(grimage_sample, grimage_stats)

        # output
        cpu_count = get_last_name(idkoiff_filename)
        if (int(cpu_count) != 1 and ((int(cpu_count) % 2) == 1)):
            continue

        output = cpu_count + ' '

        output += str(idkoiff_stats['speedup_avg']) + ' '
        output += str(idkoiff_stats['speedup_avg_low']) + ' '
        output += str(idkoiff_stats['speedup_avg_high']) + ' '

        output += str(grimage_stats['speedup_avg']) + ' '
        output += str(grimage_stats['speedup_avg_low']) + ' '
        output += str(grimage_stats['speedup_avg_high'])

        print(output)

    return


if __name__ == '__main__':
    import sys
    dirname = '../../measures/cpucount'
    main(dirname + '_idkoiff', dirname + '_grimage')
