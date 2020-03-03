# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Direct RX Gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi


class DirectRXGate(Gate):
    """Direct RX(theta) gate accomplished at the pulse level."""
    def __init__(self, theta, label=None):
        super().__init__("direct_rx_%s" % theta, 1, [theta], label=label)


def direct_rx(self, theta, q):
    """Apply direct rx(theta) to q."""
    return self.append(DirectRXGate(theta), [q], [])


QuantumCircuit.direct_rx = direct_rx
