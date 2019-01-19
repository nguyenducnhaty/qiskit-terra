# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Generic Bit object to represent either quantum or classical bits.
"""
import itertools


class Bit(object):
    """A quantum or classical bit in a quantum circuit."""

    def __init__(self, name=None):
        """Create a new generic Bit.

        Args:
            name: bit string identifier.
        """
        if name is None:
            name = '%i' % next(self.instances_counter)

        self.name = name

    def __repr__(self):
        """Return the official string representing the clbit."""
        return "%s('%s')" % (self.__class__.__qualname__, self.name)
