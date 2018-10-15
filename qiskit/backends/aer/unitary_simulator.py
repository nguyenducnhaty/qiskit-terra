# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Contains a Python simulator that returns the unitary of the circuit.

It simulates a unitary of a quantum circuit that has been compiled to run on
the simulator. It is exponential in the number of qubits.

.. code-block:: python

    UnitarySimulator().run(qobj)

Where the input is a Qobj object and the output is a AerJob object, which can
later be queried for the Result object. The result will contain a 'unitary'
data field, which is a 2**n x 2**n complex numpy array representing the
circuit's unitary matrix.

"""
import logging
import uuid
import time

import numpy as np

from qiskit.result._utils import copy_qasm_from_qobj_into_result
from qiskit.backends import BaseBackend
from qiskit.backends.aer.aerjob import AerJob
from qiskit import QISKitError
from ._simulatortools import single_gate_matrix, einsum_matmul_index

logger = logging.getLogger(__name__)


# TODO add ["status"] = 'DONE', 'ERROR' especitally for empty circuit error
# does not show up


class UnitarySimulator(BaseBackend):
    """Python implementation of a unitary simulator."""

    DEFAULT_CONFIGURATION = {
        'backend_name': 'unitary_simulator',
        'backend_version': 1.0,
        'n_qubits': -1,
        'url': 'https://github.com/QISKit/qiskit-terra',
        'simulator': True,
        'local': True,
        'conditional': False,
        'description': 'A python simulator for unitary matrix corresponding to a circuit',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u1,u2,u3,cx,id'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=configuration or self.DEFAULT_CONFIGURATION.copy(),
                         provider=provider)

        # Define attributes inside __init__.
        self._unitary_state = None
        self._number_of_qubits = 0

    def _add_unitary_single(self, gate, qubit):
        """Apply the single-qubit gate.

        gate is the single-qubit gate.
        qubit is the qubit to apply it on counts from 0 and order
            is q_{n-1} ... otimes q_1 otimes q_0.
        number_of_qubits is the number of qubits in the system.
        """
        # Convert to complex rank-2 tensor
        gate_tensor = np.array(gate, dtype=complex)
        # Compute einsum index string for 1-qubit matrix multiplication
        indexes = einsum_matmul_index([qubit], self._number_of_qubits)
        # Apply matrix multiplication
        self._unitary_state = np.einsum(indexes,
                                        gate_tensor,
                                        self._unitary_state,
                                        dtype=complex,
                                        casting='no')

    def _add_unitary_two(self, gate, qubit0, qubit1):
        """Apply the two-qubit gate.

        gate is the two-qubit gate
        qubit0 is the first qubit (control) counts from 0
        qubit1 is the second qubit (target)
        returns a complex numpy array
        """

        # Convert to complex rank-4 tensor
        gate_tensor = np.reshape(np.array(gate, dtype=complex), 4 * [2])

        # Compute einsum index string for 2-qubit matrix multiplication
        indexes = einsum_matmul_index([qubit0, qubit1], self._number_of_qubits)

        # Apply matrix multiplication
        self._unitary_state = np.einsum(indexes,
                                        gate_tensor,
                                        self._unitary_state,
                                        dtype=complex,
                                        casting='no')

    def run(self, qobj):
        """Run qobj asynchronously.

        Args:
            qobj (dict): job description

        Returns:
            AerJob: derived from BaseJob
        """
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj):
        """Run qobj. This is a blocking call.

        Args:
            job_id (str): unique id for the job.
            qobj (Qobj): job description
        Returns:
            Result: Result object
        """
        result_list = []
        start = time.time()
        for circuit in qobj.experiments:
            result_list.append(self.run_circuit(circuit))
        end = time.time()
        result = {'backend': self._configuration['name'],
                  'id': qobj.qobj_id,
                  'job_id': job_id,
                  'result': result_list,
                  'status': 'COMPLETED',
                  'success': True,
                  'time_taken': (end - start)}
        copy_qasm_from_qobj_into_result(qobj, result)

        return result_from_old_style_dict(
            result, [circuit.header.name for circuit in qobj.experiments])

    def run_circuit(self, circuit):
        """Apply the single-qubit gate.

        Args:
            circuit (QobjExperiment): experiment from qobj experiments list

        Returns:
            dict: A dictionary of results.

        Raises:
            QISKitError: if the number of qubits in the circuit is greater than 24.
            Note that the practical qubit limit is much lower than 24.
        """
        self._number_of_qubits = circuit.header.number_of_qubits
        if self._number_of_qubits > 24:
            raise QISKitError("np.einsum implementation limits unitary_simulator" +
                              " to 24 qubit circuits.")
        result = {
            'data': {},
            'name': circuit.header.name
        }

        # Initilize unitary as rank 2*N tensor
        self._unitary_state = np.reshape(np.eye(2 ** self._number_of_qubits,
                                                dtype=complex),
                                         self._number_of_qubits * [2, 2])

        for operation in circuit.instructions:
            if operation.name in ('U', 'u1', 'u2', 'u3'):
                params = getattr(operation, 'params', None)
                qubit = operation.qubits[0]
                gate = single_gate_matrix(operation.name, params)
                self._add_unitary_single(gate, qubit)
            elif operation.name in ('id', 'u0'):
                pass
            elif operation.name in ('CX', 'cx'):
                qubit0 = operation.qubits[0]
                qubit1 = operation.qubits[1]
                gate = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
                                 [0, 1, 0, 0]])
                self._add_unitary_two(gate, qubit0, qubit1)
            elif operation.name == 'measure':
                logger.info('Warning have dropped measure from unitary '
                            'simulator')
            elif operation.name == 'reset':
                logger.info('Warning have dropped reset from unitary '
                            'simulator')
            elif operation.name == 'barrier':
                pass
            else:
                result['status'] = 'ERROR'
                return result
        # Reshape unitary rank-2n tensor back to a matrix
        result['data']['unitary'] = np.reshape(self._unitary_state,
                                               2 * [2 ** self._number_of_qubits])
        result['status'] = 'DONE'
        result['success'] = True
        result['shots'] = 1
        return result
