# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Classical bit reference object.
"""
from enum import Enum


class Clbit(object):
    """Implement a qubit data type."""
    
    class ClbitType(Enum):
        REGISTER = 0
        MEMORY = 1

    def __init__(self, index, type):
        self.index = index
        self.type = type
