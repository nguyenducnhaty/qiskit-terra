# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qubit objects.
"""

class Qubit(object):
    """Implement a qubit data type."""


class PhysicalQubit(Qubit):
    """A physical qubit is a circuit qubit bound to a device."""


class VirtualQubit(Qubit):
    """A virtual qubit is a non-fault-tolerant qubit, not tied to circuit."""


class LogicalQubit(Qubit):
    """A logical qubit is a fault-tolerant qubit."""
