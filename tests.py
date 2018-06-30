from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import unittest
from collections import Counter

from weightedDict import WeightedDict


class TestWeightedDict(unittest.TestCase):
    # Usage example
    def test_main(self):
        random.seed(42)

        wdict = WeightedDict()

        wdict['dog'] = 38.2
        wdict['cat'] = 201.7
        wdict['cow'] = 222.3
        wdict['ostrich'] = 0.
        wdict['cow'] = 31.5  # Change the weight for cow
        wdict['unicorn'] = 0.01
        wdict['wolf'] = 128.1
        wdict['bear'] = 12.1
        wdict['aardvark'] = 9.1

        print(wdict['dog'])
        print(wdict.sample())
        print(wdict.keys())

        wdict.pop('cat')  # Remove the cat

        dasum = 0.
        tallies = {}
        num_samples = 100000

        for i in wdict:
            tallies[i] = 0
        for _ in range(num_samples):
            tallies[wdict.sample()] += 1
        for i in wdict:
            dasum += wdict[i]
        for i in wdict:
            print(i, tallies[i], '%.2f' % (num_samples * wdict[i] / dasum))

        print(wdict)

    # A more rigorous test
    def test_big(self):
        random.seed(42)

        dstr = 'bcdefghijklmnopqrstuvwxyz'
        data = {i: j for i, j in zip(dstr, [x + 1 for x in range(len(dstr))])}
        foo = WeightedDict()

        for i in dstr:
            foo[i] = data[i]

        # Check the sampling
        bar = Counter()
        dnum = 10000

        for _ in range(dnum):
            bar[foo.sample()] += 1

        den = sum(data.values())
        vals = {i: int(dnum * (j / den)) for i, j in data.items()}

        self.assertEqual(set(vals.keys()), set(bar.keys()))

        dsum = 0

        for i in sorted(vals):
            dif = abs(vals[i] - bar[i])
            dsum += dif
            print(i, vals[i], bar[i])

        print('Total percent from max: ' + str(100 * float(dsum) / dnum) + '%')

        self.assertLess((100 * float(dsum) / dnum), 10)

        # Check insert and deletion consistency.
        data2 = data.copy()

        for ii in range(30000):
            foo.check_tree()
            toggle = random.choice(dstr)
            print(ii, toggle, dstr)

            if toggle not in data2:
                data2[toggle] = data[toggle]
                foo[toggle] = data[toggle]
            else:
                data2.pop(toggle)
                foo.pop(toggle)

            self.assertEqual(tuple(foo.keys()), tuple(sorted(data2.keys())))

            for i, j in data2.items():
                self.assertLess(abs(foo[i] - j), .000000001)

            # Test emptying the tree
            if ii % 10000 == 0:
                dkeys = foo.keys()
                random.shuffle(dkeys)

                for toggle in dkeys:
                    foo.check_tree()
                    data2.pop(toggle)
                    foo.pop(toggle)

                    self.assertEqual(
                        tuple(foo.keys()), tuple(sorted(data2.keys())))
                    for i, j in data2.items():
                        self.assertLess(abs(foo[i] - j), .000000001)

        print(foo)
        print('Success.  Yay!')


# Note that the test output isn't identical across Python versions (2
# & 3) because random.seed has changed.  We could use version=1, but
# that's not compatible with Python 2.
if __name__ == "__main__":
    unittest.main()
