# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Classical bit objects.
"""
import itertools


class Clbit(object):
    """Implement a clbit data type."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()

    def __init__(self, name=None):
        """Create a new generic register.

        Args:
            name: clbit string identifier.
        """
        if name is None:
            name = '%i' % next(self.instances_counter)

        self.name = name

    def __repr__(self):
        """Return the official string representing the clbit."""
        return "%s('%s')" % (self.__class__.__qualname__, self.name)


class MemoryClbit(Clbit):
    """Slow classical bits in memory."""


class RegisterClbit(Clbit):
    """Fast classical bits in memory (used in feedback)."""
