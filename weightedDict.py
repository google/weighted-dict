# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# A "dictionary" for logarithmic time sampling of keys according to a
# probability distribution defined by the keys' (normalized) values.
#
# The values are assumed to be non-negative.
# The following operations are all O(log n) time, worst case:
#
# wdict.sample()        # random sample
# wdict[key] = value    # insertion (or value update)
# wdict.remove(key)     # deletion
# val = wdict[key]      # key-value lookup
# key in wdict          # key lookup
# for key in wdict: ... # iteration over keys (per iteration) O(log n) memory.
# len(wdict)            # Get length, (constant time).
#
# by Marc Pickett https://ai.google/research/people/MarcPickett

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from itertools import chain


# A red-black tree with summaries at each node.
# Only the leaves contain values for keys.
class WeightedDict:
    # key, val, sort by key
    # val is assumed to be <= 0.
    def __init__(self, min_key=None, val=0., parent=None):
        self.length = 1 if parent else 0
        self.min_key = min_key
        self.val = val
        self.lt = None
        self.rt = None
        self.up = parent
        self.black = False  # Whether the node is black.

    # Adds a new element to the list.
    def __setitem__(self, key, val):
        # If key is already in dict, remove it and add it with the new value.
        if key in self:
            self.update_val(key, float(val))
        else:
            self.add_element(key, float(val))

    # The whole purpose of this data structure!
    def sample(self):
        if not self.lt:
            return self.min_key
        if random.random() < self.lt.val / self.val:
            return self.lt.sample()
        else:
            return self.rt.sample()

    # Returns the length of the array/WeightedDict.
    def __len__(self):
        return self.length

    # Returns true iff the tree contains element.
    def __contains__(self, key):
        # See if we're at the bottom.
        if not self.lt:
            return self.min_key == key
        elif self.rt.min_key <= key:
            return key in self.rt

        return key in self.lt

    # Returns val give key.  Assumes key in tree.
    def __getitem__(self, key):
        if not self.lt:
            return self.val  # See if we're at the bottom.

        # Else, look to right, then left.
        elif self.rt.min_key <= key:
            return self.rt[key]

        return self.lt[key]

    def keys(self):
        if not self.lt:
            return [self.min_key] if self.min_key is not None else []
        return self.lt.keys() + self.rt.keys()

    # Iterates over the keys in the keys' sorted order.
    def __iter__(self):
        if not self.lt:
            if self.min_key is not None:
                yield self.min_key

        else:
            for n in chain(self.lt, self.rt):
                yield n

    # Deletes an element by key.
    def remove(self, index):
        if self.del_element(index):
            self.min_key, self.val = None, 0.

    def pop(self, index):
        dval = self[index]

        self.remove(index)
        return dval

    # "Private" methods
    # Assumes key is already in the dict
    def update_val(self, key, newval):
        if not self.lt:
            self.val = newval
            if self.up:
                self.up.set_vals()

        elif self.rt.min_key <= key:
            self.rt.update_val(key, newval)

        else:
            self.lt.update_val(key, newval)

    # Fixes the val for all ancestors
    def set_vals(self):
        self.val = self.lt.val + self.rt.val
        if self.up:
            self.up.set_vals()

    def add_element(self, key, val):
        self.length += 1

        # Check to see if we're the top.

        if self.min_key is None:
            self.min_key, self.val = key, val
        # See if we're at the bottom.
        elif not self.lt:
            self.split(key, val)
        else:
            self.val += val
            # Look to the right.
            if self.rt.min_key <= key:
                self.rt.add_element(key, val)
            # Look to the left.
            else:
                if self.min_key > key:
                    self.min_key = key
                self.lt.add_element(key, val)

    # Splits the WeightedDict into 2.
    def split(self, key, val):
        # Make right the larger of the 2.
        (lkey, lval), (rkey, rval) = sorted(
            [(self.min_key, self.val), (key, val)])

        self.val = lval + rval
        self.min_key = lkey

        self.lt = WeightedDict(lkey, lval, self)
        self.rt = WeightedDict(rkey, rval, self)

        # Balancing the lt also takes care of the right.
        self.lt.rb_balance()

    # This does the balancing for when a node is added
    def rb_balance(self):
        # If we're the top do nothing or the parent is black, do nothing.
        if not self.up or self.up.black:
            pass

        # If we're 1 below the top.
        elif not self.up.up:
            self.up.black = True

        # If our uncle's red then push the black grandparent down and recurse.
        elif not self.up.brother().black:
            self.up.up.black = False
            self.up.black = True
            self.up.brother().black = True
            self.up.up.rb_balance()

        elif not self.brother().black:  # If uncle's black & brother's red.
            grandpa = self.up.up
            self.up.black = True

            if grandpa.lt == self.up:  # If our parent is on the left side
                self.up.lt.black = True
                self.up.up.rshift()
            else:  # If our parent is on the right side
                self.up.rt.black = True
                self.up.up.lshift()

            grandpa.black = False
            grandpa.rb_balance()

        else:  # If uncle's black & brother's black.
            # If we're left and our parent's left.
            if self.up.lt == self and self.up.up.lt == self.up:
                self.up.up.rshift()

            # If we're right and our parent's right.
            elif self.up.rt == self and self.up.up.rt == self.up:
                self.up.up.lshift()

            elif self.up.rt == self:  # If we're right.
                self.up.lshift()
                self.up.up.rshift()

            else:  # If we're left.
                self.up.rshift()
                self.up.up.lshift()

    # Returns the brother of this node.
    def brother(self):
        return self.up.rt if self.up.lt == self else self.up.lt

    # Perform these operations:
    #         rshift                        lshift
    #      ^    -->    ^                ^    <--    ^
    #     / \   -->   / \              / \   <--   / \
    #    X   C  -->  A   X            X   C  <--  A   X
    #   / \     -->     / \          / \     <--     / \
    #  A   B    -->    B   C        A   B    <--    B   C
    def rshift(self):
        a = self.lt.lt
        b = self.lt.rt
        c = self.rt
        x = self.lt

        self.lt, self.rt = a, x

        x.length = len(b) + len(c)
        x.val = b.val + c.val
        x.min_key = b.min_key
        x.lt, x.rt = b, c
        a.up, c.up = self, x  # Establish heredity

    def lshift(self):
        b = self.rt.lt
        c = self.rt.rt
        x = self.rt
        a = self.lt

        self.lt, self.rt = x, c

        x.length = len(a) + len(b)
        x.val = a.val + b.val
        x.min_key = a.min_key
        x.lt, x.rt = a, b
        c.up, a.up = self, x  # Establish heredity

    def del_element(self, key):
        self.length -= 1

        # See if we're at the bottom.
        if not self.lt:
            return True

        if self.rt.min_key <= key:  # Look to the right.
            if self.rt.del_element(key):
                self.unsplit(self.lt)

        # Look to the left.
        elif self.lt.del_element(key):
            self.unsplit(self.rt)

        return False

    # Hook the node to the W's l and r.
    #      T    -->    T            T      -->    T
    #     / \   -->   / \          / \     -->   / \
    #    W   X  -->  A   B   or   X   W    -->  A   B
    #   / \     -->                  / \   -->
    #  A   B    -->                 A   B  -->
    def unsplit(self, w):
        self.lt, self.rt, self.min_key, self.val = w.lt, w.rt, w.min_key, w.val

        # Tell the trees of their new parent if they're not None.
        if self.lt:
            self.lt.up, self.rt.up = self, self
        if self.up:
            self.up.set_mins()

        self.rb_unsplit_fix(w.black)

    # Propogates the changes to ancestors
    def set_mins(self):
        self.val = self.lt.val + self.rt.val
        self.min_key = self.lt.min_key

        if self.up:
            self.up.set_mins()

    # Tell the children that you're their parent, then fix the rb properties.
    def rb_unsplit_fix(self, nuked_black):
        if not self.up:
            self.black = True  # See if we're at the top.

        elif not nuked_black:
            pass  # See if the nuked node was red.

        elif not self.black:
            self.black = True  # See if we're red.

        else:
            self.rb_solve_double_black()  # We have a double black node.

    # This is for when a node counts as 2 blacks(or black when it's red).
    def rb_solve_double_black(self):
        # See if we're red.
        if not self.black:
            self.black = True

        elif not self.up:
            pass  # Do nothing if we're at the top..

        elif not self.brother().black:  # See if our brother's red.
            # Figure out to shift left or right.
            if self.up.lt == self:
                self.up.lshift()
            else:
                self.up.rshift()
            self.rb_solve_double_black()

        # See if our brother's children are both black.
        elif self.brother().lt.black and self.brother().rt.black:
            self.brother().black = False
            self.up.rb_solve_double_black()

        elif self.up.lt == self:  # See if we're on the left side.
            if not self.up.rt.rt.black:  # See if our furthest nephew is red.
                self.up.rt.rt.black = True
                self.up.lshift()
            else:  # Our nearest nephew must be red.
                self.up.rt.rshift()
                self.rb_solve_double_black()

        else:  # OK, we're on the right side.
            if not self.up.lt.lt.black:  # See if our furthest nephew is red.
                self.up.lt.lt.black = True
                self.up.rshift()
            else:  # Our nearest nephew must be red.
                self.up.lt.lshift()
                self.rb_solve_double_black()

    # For testing
    def __str__(self):
        picture = [[' '] * (self.length * 4) for _ in range(self.depth() + 1)]
        centre = self.get_str(picture, 0)
        dstr = ' ' * centre + '|\n'

        for i in reversed(picture):
            dstr += ''.join(i).rstrip() + '\n'
        return dstr

    # Max nodes to bottom.
    def depth(self):
        return 0 if not self.lt else\
            (1 + max(self.lt.depth(), self.rt.depth()))

    # Returns the centre x-coordinate of this node.
    def get_str(self, picture, offset):
        depth = self.depth()

        if depth == 0:
            picture[0][4 * offset] = '*' if self.black else 'o'
            return offset * 4

        l_centre = self.lt.get_str(picture, offset)
        r_centre = self.rt.get_str(picture, offset + len(self.lt))
        my_centre = (l_centre + r_centre) // 2

        for i in range(l_centre, r_centre, 1):
            picture[depth][i] = '-'
        for i in range(self.lt.depth() + 1, depth):
            picture[i][l_centre] = '|'
        for i in range(self.rt.depth() + 1, depth):
            picture[i][r_centre] = '|'

        picture[depth][my_centre] = '*' if self.black else 'o'
        picture[depth][l_centre] = '/'
        picture[depth][r_centre] = '\\'

        return my_centre

    def check_tree(self):
        if self.lt:
            assert (self.lt.up == self)
            assert (self.rt.up == self)
            assert (self.min_key == self.lt.min_key)
            assert (self.val == self.lt.val + self.rt.val)

            # Both children of every red node should be black (or None)
            assert (self.black or (self.lt.black and self.rt.black))

            self.lt.check_tree()
            self.rt.check_tree()
