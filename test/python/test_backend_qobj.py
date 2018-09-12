# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Integration tests for sending Qobj and receving Result from backends."""

from qiskit.backends.ibmq import IBMQProvider
from .common import Path, QiskitTestCase, requires_qe_access, slow_test


class TestBackendQobj(QiskitTestCase):
    """Qiskit backend qobj test."""

    @slow_test
    @requires_qe_access
    def test_simple_circuit(self, qe_token, qe_url):
        """Test one circuit, one register, in-order readout.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])
        local = get_backend('local_qasm_simulator')
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        for remote in remotes:
            self.log.info(remote.status())
            if (remote.configuration()['operational'] and
                remote.configuration()['allow_q_object']):
                qobj = compile(circ, remote)
                result_remote = remote.run(qobj).result()
                result_local = local.run(qobj).result()
                assertDictAlmostEqual(result_remote.get_counts(circ),
                                      result_local.get_counts(circ))

    @slow_test
    @requires_qe_access
    def test_readout_order(self, qe_token, qe_url):
        """Test one circuit, one register, out-of-order readout.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.x(qr[2])
        circ.measure(qr[0], cr[2])
        circ.measure(qr[1], cr[0])
        circ.measure(qr[2], cr[1])
        circ.measure(qr[3], cr[3])
        local = get_backend('local_qasm_simulator')
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        for remote in remotes:
            self.log.info(remote.status())
            if (remote.configuration()['operational'] and
                remote.configuration()['allow_q_object']):
                qobj = compile(circ, remote)
                result_remote = remote.run(qobj).result()
                result_local = local.run(qobj).result()
                assertDictAlmostEqual(result_remote.get_counts(circ),
                                      result_local.get_counts(circ))

    @slow_test
    @requires_qe_access
    def test_multi_register(self, qe_token, qe_url):
        """Test one circuit, two registers, out-of-order readout.
        """
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(1)
        circ = QuantumCircuit(qr1, qr2, cr1, cr2)
        circ.h(qr1[0])
        circ.cx(qr1[0], qr2[1])
        circ.h(qr2[0])
        circ.h(qr2[0], qr1[3])
        circ.x(qr1[1])
        circ.measure(qr1[0], cr2[1])
        circ.measure(qr1[1], cr1[0])
        circ.measure(qr1[2], cr2[0])
        circ.measure(qr1[3], cr1[3])
        circ.measure(qr2[0], cr1[2])
        circ.measure(qr2[1], cr1[1])
        local = get_backend('local_qasm_simulator')
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        for remote in remotes:
            self.log.info(remote.status())
            if (remote.configuration()['operational'] and
                remote.configuration()['allow_q_object']):
                qobj = compile(circ, remote)
                result_remote = remote.run(qobj).result()
                result_local = local.run(qobj).result()
                assertDictAlmostEqual(result_remote.get_counts(circ),
                                      result_local.get_counts(circ))

    @slow_test
    @requires_qe_access
    def test_multi_circuit(self, qe_token, qe_url):
        """Test one circuit, two registers, out-of-order readout.
        """
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(1)
        circ1 = QuantumCircuit(qr1, qr2, cr1, cr2)
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr2[1])
        circ1.h(qr2[0])
        circ1.h(qr2[0], qr1[3])
        circ1.x(qr1[1])
        circ1.measure(qr1[0], cr2[1])
        circ1.measure(qr1[1], cr1[0])
        circ1.measure(qr1[2], cr2[0])
        circ1.measure(qr1[3], cr1[3])
        circ1.measure(qr2[0], cr1[2])
        circ1.measure(qr2[1], cr1[1])
        circ2 = QuantumCircuit(qr1, qr2, cr1)
        circ2.h(qr1[0])
        circ2.cx(qr1[0], qr1[3])
        circ2.h(qr2[1])
        circ2.h(qr2[1], qr1[3])
        circ2.measure(qr1[0], cr1[0])
        circ2.measure(qr1[1], cr1[1])
        circ2.measure(qr1[2], cr1[2])
        circ2.measure(qr2[1], cr1[3])
        local = get_backend('local_qasm_simulator')
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        for remote in remotes:
            self.log.info(remote.status())
            if (remote.configuration()['operational'] and
                remote.configuration()['allow_q_object']):
                qobj = compile([circ1, circ2], remote)
                result_remote = remote.run(qobj).result()
                result_local = local.run(qobj).result()
                assertDictAlmostEqual(result_remote.get_counts(circ1),
                                      result_local.get_counts(circ1))
                assertDictAlmostEqual(result_remote.get_counts(circ2),
                                      result_local.get_counts(circ2))
