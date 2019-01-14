# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum bit reference object.
"""
from enum import Enum


class Qubit(object):
    """Implement a qubit data type."""
    
    class QubitType(Enum):
        PHYSICAL = 0
        VIRTUAL = 1
        LOGICAL = 2

    def __init__(self, index, type):
        self.index = index
        self.type = type
