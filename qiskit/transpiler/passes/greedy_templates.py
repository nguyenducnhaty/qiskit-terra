# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Pass for template-based optimizations.
"""
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import CnotGate
from qiskit.transpiler.basepasses import TransformationPass


class GreedyTemplates(TransformationPass):
    """Find simple pre-defined template equivalents and apply them."""

    templates = dict()

    """template 1"""
    q = QuantumRegister(3)
    dag1 = DAGCircuit()
    dag1.add_qreg(q)
    dag1.apply_operation_back(CnotGate(q[0], q[2]), [q[0], q[2]])
    dag1.apply_operation_back(CnotGate(q[1], q[2]), [q[1], q[2]])

    dag2 = DAGCircuit()
    dag2.add_qreg(q)
    dag2.apply_operation_back(CnotGate(q[1], q[2]), [q[1], q[2]])
    dag2.apply_operation_back(CnotGate(q[0], q[2]), [q[0], q[2]])

    templates[dag1] = dag2

    """template 2"""
    q = QuantumRegister(2)
    dag1 = DAGCircuit()
    dag1.add_qreg(q)
    dag1.apply_operation_back(CnotGate(q[0], q[1]), [q[0], q[1]])
    dag1.apply_operation_back(CnotGate(q[1], q[0]), [q[1], q[0]])
    dag1.apply_operation_back(CnotGate(q[0], q[1]), [q[0], q[1]])
    dag1.apply_operation_back(CnotGate(q[1], q[0]), [q[1], q[0]])

    q = QuantumRegister(2)
    dag2 = DAGCircuit()
    dag2.add_qreg(q)
    dag2.apply_operation_back(CnotGate(q[1], q[0]), [q[1], q[0]])
    dag2.apply_operation_back(CnotGate(q[0], q[1]), [q[0], q[1]])

    templates[dag1] = dag2

    def run(self, dag):
        """
        Run one pass of applying templates in a greedy fashion.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
        Returns:
            DAGCircuit: Transformed DAG.
        """
        for node in dag.op_nodes():
            if isinstance(node.op, CnotGate):
                for d in dag.descendants(node):
                    if d.type == 'op' and isinstance(d.op, CnotGate) and \
                        d.qargs[1] == node.qargs[1]:
                            pass

        return dag
