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
Two-pulse single-qubit gate.
"""

import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit


class U3Gate(Gate):
    """Two-pulse single-qubit gate."""

    def __init__(self, theta, phi, lam, label=None):
        """Generic single-qubit rotation gate.

        .. math::

        U2(phi, lam) = [[cos(theta/2), (e^-i.lam).sin(theta/2)],
                        [(e^i.phi).sin(theta/2), e^i(phi+lam)cos(theta/2)]]

        Implemented using two X90 pulse on IBM Q systems:

        .. math::

            U3(theta, phi, lam) = RZ(phi+pi).RX(90).RZ(theta+pi).RX(90).RZ(lam)
        """
        super().__init__("u3", 1, [theta, phi, lam], label=label)

    def inverse(self):
        """Invert this gate.

        .. math::
        
        U3(theta, phi, lam)^dagger = U3(-theta, -lam, -phi)
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        return numpy.array(
            [[
                numpy.cos(theta / 2),
                -numpy.exp(1j * lam) * numpy.sin(theta / 2)
            ],
             [
                 numpy.exp(1j * phi) * numpy.sin(theta / 2),
                 numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2)
             ]],
            dtype=complex)


def u3(self, theta, phi, lam, q):  # pylint: disable=invalid-name
    """Apply u3 to q."""
    return self.append(U3Gate(theta, phi, lam), [q], [])


QuantumCircuit.u3 = u3
