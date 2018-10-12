# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Qobj utilities and enums."""

import operator
from itertools import groupby
from enum import Enum


class QobjType(str, Enum):
    """
    Qobj.type allowed values.
    """
    QASM = 'QASM'
    PULSE = 'PULSE'


def get_register_sizes(bit_labels):
    """
    Get the size of unique registers from bit_labels list.

    Args:
        bit_labels (list): this list is of the form::

            [['reg1', 0], ['reg1', 1], ['reg2', 0]]

            which indicates a register named "reg1" of size 2
            and a register named "reg2" of size 1. This is the
            format of classic and quantum bit labels in qobj
            header.

    Yields:
        tuple: iterator of register_name:size pairs.
    """
    it = groupby(bit_labels, operator.itemgetter(0))
    for register_name, sub_it in it:
        yield register_name, max(ind[1] for ind in sub_it) + 1 
