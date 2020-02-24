# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
"""
Decompose a single-qubit unitary via Euler angles.
"""

import math
import numpy as np
import scipy.linalg as la

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (U3Gate, U1Gate, RXGate, RYGate, RZGate,
                                                   RGate)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

DEFAULT_ATOL = 1e-12


class OneQubitEulerDecomposer:
    r"""A class for decomposing 1-qubit unitaries into Euler angle rotations.

    The resulting decomposition is parameterized by 3 Euler rotation angle
    parameters :math:`(\theta, \phi, \lambda)`, and a phase parameter
    :math:`\gamma`. The value of the parameters for an input unitary depends
    on the decomposition basis. Allowed bases and the resulting circuits are
    shown in the following table. Note that for the non-Euler bases (U3, U1X,
    RR), the ZYZ euler parameters are used.

    .. list-table:: Supported circuit bases
        :widths: auto
        :header-rows: 1

        * - Basis
          - Euler Angle Basis
          - Decomposition Circuit
        * - 'ZYZ'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R_Z(\phi).R_Y(\theta).R_Z(\lambda)`
        * - 'ZXZ'
          - :math:`Z(\phi) X(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R_Z(\phi).R_X(\theta).R_Z(\lambda)`
        * - 'XYX'
          - :math:`X(\phi) Y(\theta) X(\lambda)`
          - :math:`e^{i\gamma} R_X(\phi).R_Y(\theta).R_X(\lambda)`
        * - 'U3'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma}{2}\right)\right)} U_3(\theta,\phi,\lambda)`
        * - 'U1X'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i \gamma} U_1(\phi+\pi).R_X\left(\frac{\pi}{2}\right).`
            :math:`U_1(\theta+\pi).R_X\left(\frac{\pi}{2}\right).U_1(\lambda)`
        * - 'RR'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R\left(-\pi,\frac{\phi-\lambda+\pi}{2}\right).`
            :math:`R\left(\theta+\pi,\frac{\pi}{2}-\lambda\right)`
    """

    def __init__(self, basis='U3', up_to_diagonal=False):
        """Initialize decomposer

        Supported bases are: 'U3', 'U1X', 'RR', 'ZYZ', 'ZXZ', 'XYX'.

        Setting ``up_to_diagonal = True`` returns a possibly more optimized
        decomposition that is equivalent up to a left-multiplied diagonal
        (i.e. equivalent up to a relative phase). That is, u=D.u', where u'
        is the unitary implemented by the found circuit. The residual diagonal
        can be obtained using the ``get_diagonal()`` method.

        Args:
            basis (str): the decomposition basis [Default: 'U3']

        Raises:
            QiskitError: If input basis is not recognized.
        """
        if basis not in ['U3', 'U1X', 'ZYZ', 'ZXZ', 'XYX']:
            raise QiskitError("OneQubitEulerDecomposer: unsupported basis")
        self._basis = basis
        self._diag = [1., 1.]
        self._up_to_diagonal = up_to_diagonal

    def __call__(self,
                 unitary,
                 simplify=True,
                 atol=DEFAULT_ATOL):
        """Decompose single qubit gate into a circuit.

        Args:
            unitary (Operator or Gate or array): 1-qubit unitary matrix
            simplify (bool): reduce gate count in decomposition [Default: True].
            atol (bool): absolute tolerance for checking angles when simplifing
                         returnd circuit [Default: 1e-12].

        Returns:
            QuantumCircuit: the decomposed single-qubit gate circuit
            array: the residual diagonal if ``up_to_diagonal`` is True

        Raises:
            QiskitError: if input is invalid or synthesis fails.
        """
        if hasattr(unitary, 'to_operator'):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            unitary = unitary.to_operator().data
        elif hasattr(unitary, 'to_matrix'):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            unitary = unitary.to_matrix()
        # Convert to numpy array incase not already an array
        unitary = np.asarray(unitary, dtype=complex)

        # Check input is a 2-qubit unitary
        if unitary.shape != (2, 2):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "expected 2x2 input matrix")
        if not is_unitary_matrix(unitary):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "input matrix is not unitary.")
        theta, phi, lam, _ = self._params(unitary)
        circuit = self._circuit(theta, phi, lam,
                                simplify=simplify,
                                atol=atol)
        return circuit

    @property
    def basis(self):
        """The decomposition basis."""
        return self._basis

    def get_diagonal(self):
        """Return the diagonal up to which the unitary was synthesized."""
        return self._diag

    def _angles(self, unitary_mat):
        """Return Euler angles for given basis."""
        if self._basis in ['U3', 'U1X', 'ZYZ']:
            return self._angles_zyz(unitary_mat)
        if self._basis == 'ZXZ':
            return self._angles_zxz(unitary_mat)
        if self._basis == 'XYX':
            return self._angles_xyx(unitary_mat)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    def _circuit(self, unitary_mat, simplify=True, up_to_diagonal=False,
                 atol=DEFAULT_ATOL):
        # NOTE: The 4th variable is phase to be used later
        theta, phi, lam, _ = self._angles(unitary_mat)
        if self._basis == 'U3':
            return self._circuit_u3(theta, phi, lam)
        if self._basis == 'U1X':
            return self._circuit_u1x(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        if self._basis == 'ZYZ':
            return self._circuit_zyz(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        if self._basis == 'ZXZ':
            return self._circuit_zxz(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        if self._basis == 'XYX':
            return self._circuit_xyx(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    @basis.setter
    def basis(self, basis):
        """Set the decomposition basis."""
        basis_methods = {
            'U3': (self._params_u3, self._circuit_u3),
            'U1X': (self._params_u1x, self._circuit_u1x),
            'RR': (self._params_zyz, self._circuit_rr),
            'ZYZ': (self._params_zyz, self._circuit_zyz),
            'ZXZ': (self._params_zxz, self._circuit_zxz),
            'XYX': (self._params_xyx, self._circuit_xyx)
        }
        if basis not in basis_methods:
            raise QiskitError("OneQubitEulerDecomposer: unsupported basis {}".format(basis))
        self._basis = basis
        self._params, self._circuit = basis_methods[self._basis]

    def angles(self, unitary):
        """Return the Euler angles for input array.

        Args:
            unitary (np.ndarray): 2x2 unitary matrix.

        Returns:
            tuple: (theta, phi, lambda).
        """
        theta, phi, lam, _ = self._params(unitary)
        return theta, phi, lam

    def angles_and_phase(self, unitary):
        """Return the Euler angles and phase for input array.

        Args:
            unitary (np.ndarray): 2x2 unitary matrix.

        Returns:
            tuple: (theta, phi, lambda, phase).
        """
        return self._params(unitary)

    @staticmethod
    def _params_zyz(mat):
        """Return the euler angles and phase for the ZYZ basis."""
        # We rescale the input matrix to be special unitary (det(U) = 1)
        # This ensures that the quaternion representation is real
        coeff = la.det(mat)**(-0.5)
        phase = -np.angle(coeff)
        su_mat = coeff * mat  # U in SU(2)
        # OpenQASM SU(2) parameterization:
        # U[0, 0] = exp(-i(phi+lambda)/2) * cos(theta/2)
        # U[0, 1] = -exp(-i(phi-lambda)/2) * sin(theta/2)
        # U[1, 0] = exp(i(phi-lambda)/2) * sin(theta/2)
        # U[1, 1] = exp(i(phi+lambda)/2) * cos(theta/2)
        theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
        phiplambda = 2 * np.angle(su_mat[1, 1])
        phimlambda = 2 * np.angle(su_mat[1, 0])
        phi = (phiplambda + phimlambda) / 2.0
        lam = (phiplambda - phimlambda) / 2.0
        return theta, phi, lam, phase

    @staticmethod
    def _params_zxz(mat):
        """Return the euler angles and phase for the ZXZ basis."""
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat)
        return theta, phi + np.pi / 2, lam - np.pi / 2, phase

    @staticmethod
    def _params_xyx(mat):
        """Return the euler angles and phase for the XYX basis."""
        # We use the fact that
        # Rx(a).Ry(b).Rx(c) = H.Rz(a).Ry(-b).Rz(c).H
        mat_zyz = 0.5 * np.array(
            [[
                mat[0, 0] + mat[0, 1] + mat[1, 0] + mat[1, 1],
                mat[0, 0] - mat[0, 1] + mat[1, 0] - mat[1, 1]
            ],
             [
                 mat[0, 0] + mat[0, 1] - mat[1, 0] - mat[1, 1],
                 mat[0, 0] - mat[0, 1] - mat[1, 0] + mat[1, 1]
             ]],
            dtype=complex)
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat_zyz)
        return -theta, phi, lam, phase

    @staticmethod
    def _params_u3(mat):
        """Return the euler angles and phase for the U3 basis."""
        # The determinant of U3 gate depends on its params
        # via det(u3(theta, phi, lam)) = exp(1j*(phi+lam))
        # Since the phase is wrt to a SU matrix we must rescale
        # phase to correct this
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat)
        return theta, phi, lam, phase - 0.5 * (phi + lam)

    @staticmethod
    def _params_u1x(mat):
        """Return the euler angles and phase for the U1X basis."""
        # The determinant of this decomposition depends on its params
        # Since the phase is wrt to a SU matrix we must rescale
        # phase to correct this
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat)
        return theta, phi, lam, phase - 0.5 * (theta + phi + lam)

    def _circuit_u3(self, theta, phi, lam):
        circuit = QuantumCircuit(1)
        circuit.append(U3Gate(theta, phi, lam), [0])
        return circuit

    def _circuit_u1x(self, theta, phi, lam, simplify=True, atol=DEFAULT_ATOL):
        # Check for U1 and U2 decompositions into minimimal
        # required X90 pulses
        if simplify and np.allclose([theta, phi], [0., 0.], atol=atol):
            # zero X90 gate decomposition
            circuit = QuantumCircuit(1)
            if self._up_to_diagonal:
                self._diag = [1, np.exp(1j * lam)]
            else:
                circuit.append(U1Gate(lam), [0])
            return circuit
        if simplify and np.isclose(theta, np.pi / 2, atol=atol):
            # single X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.append(U1Gate(lam - np.pi / 2), [0])
            circuit.append(RXGate(np.pi / 2), [0])
            if self._up_to_diagonal:
                self._diag = [1, np.exp(1j * (phi + np.pi / 2.))]
            else:
                circuit.append(U1Gate(phi + np.pi / 2.), [0])
            return circuit
        # General two-X90 gate decomposition
        circuit = QuantumCircuit(1)
        circuit.append(U1Gate(lam), [0])
        circuit.append(RXGate(np.pi / 2), [0])
        circuit.append(U1Gate(theta + np.pi), [0])
        circuit.append(RXGate(np.pi / 2), [0])
        if self._up_to_diagonal:
            self._diag = [1, np.exp(1j * (phi + np.pi))]
        else:
            circuit.append(U1Gate(phi + np.pi), [0])
        return circuit

    def _circuit_zyz(self, theta, phi, lam, simplify=True, atol=DEFAULT_ATOL):
        circuit = QuantumCircuit(1)
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit.append(RZGate(phi + lam), [0])
            return circuit
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.append(RZGate(lam), [0])
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.append(RYGate(theta), [0])
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            if self._up_to_diagonal:
                self._diag = [np.exp(-1j * phi / 2.), np.exp(1j * phi / 2.)]
            else:
                circuit.append(RZGate(phi), [0])
        return circuit

    def _circuit_zxz(self, theta, phi, lam, simplify=False, atol=DEFAULT_ATOL):
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit = QuantumCircuit(1)
            circuit.append(RZGate(phi + lam), [0])
            return circuit
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.append(RZGate(lam), [0])
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.append(RXGate(theta), [0])
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            if self._up_to_diagonal:
                self._diag = [np.exp(-1j * phi / 2.), np.exp(1j * phi / 2.)]
            else:
                circuit.append(RZGate(phi), [0])
        return circuit

    def _circuit_xyx(self, theta, phi, lam, simplify=True, atol=DEFAULT_ATOL):
        circuit = QuantumCircuit(1)
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit.append(RXGate(phi + lam), [0])
            return circuit
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.append(RXGate(lam), [0])
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.append(RYGate(theta), [0])
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.append(RXGate(phi), [0])
        return circuit

    @staticmethod
    def _circuit_u1x(theta,
                     phi,
                     lam,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        # Shift theta and phi so decomposition is
        # U1(phi).X90.U1(theta).X90.U1(lam)
        theta += np.pi
        phi += np.pi
        # Check for decomposition into minimimal number required X90 pulses
        if simplify and np.isclose(abs(theta), np.pi, atol=atol):
            # Zero X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.append(U1Gate(lam + phi + theta), [0])
            return circuit
        if simplify and np.isclose(abs(theta), np.pi/2, atol=atol):
            # Single X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.append(U1Gate(lam + theta), [0])
            circuit.append(RXGate(np.pi / 2), [0])
            circuit.append(U1Gate(phi + theta), [0])
            return circuit
        # General two-X90 gate decomposition
        circuit = QuantumCircuit(1)
        circuit.append(U1Gate(lam), [0])
        circuit.append(RXGate(np.pi / 2), [0])
        circuit.append(U1Gate(theta), [0])
        circuit.append(RXGate(np.pi / 2), [0])
        circuit.append(U1Gate(phi), [0])
        return circuit

    @staticmethod
    def _circuit_rr(theta,
                    phi,
                    lam,
                    simplify=True,
                    atol=DEFAULT_ATOL):
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(theta, -np.pi, atol=atol):
            circuit.append(RGate(theta + np.pi, np.pi / 2 - lam), [0])
        circuit.append(RGate(-np.pi, 0.5 * (phi - lam + np.pi)), [0])
        return circuit
