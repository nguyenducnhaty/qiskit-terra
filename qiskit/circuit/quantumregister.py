# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum register reference object.
"""
import itertools

from .qubit import Qubit
from .register import Register
from qiskit.exceptions import QiskitError


class QuantumRegister(Register):
    """Implement a quantum register."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()

    # Prefix to use for auto naming.
    prefix = 'q'

    def __init__(self, size, name=None):
        """Create a new generic register.

        Args:
            size (int): a register size to create
            name: register string name
        """
        qubits = [Qubit()] * size
        super().__init__(qubits, name)

    def qasm(self):
        """Return OPENQASM string for this register."""
        return "qreg %s[%d];" % (self.name, self.size)
