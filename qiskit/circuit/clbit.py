# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Classical bit objects.
"""

class Clbit(object):
    """Implement a clbit data type."""

    def __init__(self, index):
        self.index = index

class MemoryClbit(object):
    """Slow classical bits in memory."""


class RegisterClbit(object):
    """Fast classical bits in memory (used in feedback)."""
