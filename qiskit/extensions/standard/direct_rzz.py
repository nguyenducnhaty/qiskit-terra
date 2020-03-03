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
ZZ-Interaction (e^{-i ZZ theta}) gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.cr import CRGate
from qiskit.extensions.standard.rx import RXGate
from qiskit.extensions.standard.direct_rx import DirectRXGate


class DirectRZZGate(Gate):
    """ZZ-Interaction (e^{-i ZZ theta}) gate."""

    def __init__(self, theta):
        super().__init__("direct_rzz", 2, [theta])

    def _define(self):
        """
        Decomposition into a single CR gate is
        ----X---|  CR  |-------
        ----H---|-theta|---H---
        """
        definition = []
        q = QuantumRegister(2, "q")
        theta = self.params[0]

        rule = [
            (RXGate(pi), [q[0]], []),
            (U2Gate(0, pi), [q[1]], []),
            (CRGate(-theta), [q[0], q[1]], []),
            (U2Gate(0, pi), [q[1]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition


def direct_rzz(self, theta, q1, q2):
    # NOTE: CR is implemented "one way" between each connected qubit pair.
    # ZZ interaction tries to do CR with q1 as control and q2 as control. However, if
    # we have the "wrong" direction, there is a simple fix: just flip our treatment of q1 and q2.
    # This only works because the ZZ interaction unitary is diagonal, so it is insensitive
    # to a flipping of the treatment of q1 and q2. The map of cr directions is obtained as:
    # cr_map = set()
    # for i, j in config.coupling_map:
    #     duration_ij = cmd_def.get('cx', qubits=[i, j]).duration
    #     duration_ji = cmd_def.get('cx', qubits=[j, i]).duration
    #     # shorter pulse is the one with the "native" CR direction
    #     if duration_ij < duration_ji:
    #         cr_map.add((i, j))
    #     else:
    #         cr_map.add((j, i))
    # print(cr_map)
    cr_map = {(11, 16), (10, 11), (5, 6), (2, 1), (8, 9), (1, 6), (14, 9), (16, 17), (19, 18), (6, 7), (12, 13), (16, 15), (18, 13), (12, 7), (18, 17), (2, 3), (8, 7), (11, 12), (0, 1), (10, 5), (13, 14), (3, 8), (4, 3)}

    if (q1, q2) in cr_map:  # correct CR direction
        return self.append(DirectRZZGate(theta), [q1, q2], [])
    else:  # incorrect CR, so just flip the treatment of the two qubits
        return self.append(DirectRZZGate(theta), [q2, q1], [])


QuantumCircuit.direct_rzz = direct_rzz
