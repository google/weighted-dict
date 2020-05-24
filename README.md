# WeightedDict
[![GitHub issues](https://img.shields.io/github/issues/google/weighted-dict?logo=github&color=red)](https://github.com/google/weighted-dict/issues)
[![GitHub forks](https://img.shields.io/github/forks/google/weighted-dict?logo=github&color=blue)](https://github.com/google/weighted-dict/network)
[![GitHub stars](https://img.shields.io/github/stars/google/weighted-dict?logo=github&color=orange)](https://github.com/google/weighted-dict/stargazers)
[![GitHub license](https://img.shields.io/github/license/google/weighted-dict?logo=github)](https://github.com/google/weighted-dict/blob/master/LICENSE)

A "dictionary" for logarithmic time sampling of keys according to a probability
distribution defined by the keys' (normalized) values.

## Operations

The values in the weightedDict are assumed to be non-negative.
The following operations are all worst case O(log n) time:

``` python
wdict.sample()        # Randomly sample from dict by the normalized weight. 
wdict[key] = value    # insertion (or value update)
wdict.remove(key)     # deletion
val = wdict[key]      # key-value lookup
key in wdict          # key lookup
for key in wdict: ... # iterate over keys in order.  O(log n) memory.
len(wdict)            # Get length, (constant time).
```

This works with both Python 2 or Python 3.  To test this:
``` shell
python weightedDict.py
```

Example usage:
``` python
from weightedDict import WeightedDict
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
random.seed(42)
wdict = WeightedDict()
wdict['dog'] = 38.2
wdict['cat'] = 201.7
wdict['cow'] = 222.3
wdict['ostrich'] = 0.
wdict['cow'] = 31.5 # Change the weight for cow
wdict['unicorn'] = 0.01
wdict['wolf'] = 128.1
wdict['bear'] = 12.1
wdict['aardvark'] = 9.1
print(wdict['dog'])
print(wdict.sample())
print(wdict.keys())
wdict.pop('cat') # Remove the cat
dasum = 0.
tallies = {}
numSamples = 100000
for i in wdict: tallies[i] = 0
for _ in range(numSamples): tallies[wdict.sample()] += 1
for i in wdict: dasum += wdict[i]
for i in wdict: print(i, tallies[i], '%.2f'%(numSamples * wdict[i]/dasum))
print(wdict)
```

Output should be:
```
38.2
cat
['aardvark', 'bear', 'cat', 'cow', 'dog', 'ostrich', 'unicorn', 'wolf']
aardvark 4084 4155.06
bear 5448 5524.86
cow 14441 14382.90
dog 17336 17442.13
ostrich 0 0.00
unicorn 4 4.57
wolf 58687 58490.48
            |
      /-----o------\
  /---*---\     /--*--\
/-*-\   /-o-\   |   /-*-\
o   o   *   *   *   o   o
```
