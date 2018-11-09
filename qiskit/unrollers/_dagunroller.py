# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
DAG Unroller
"""

import networkx as nx

from qiskit.unrollers._unroller import Unroller
from qiskit.qasm._node import Real, Id, IdList, ExpressionList, Gate, \
                              PrimaryList, Int, IndexedId, Qreg, If, Creg, \
                              Program, CustomUnitary
from ._unrollererror import UnrollerError
from ._dagbackend import DAGBackend


class DagUnroller(object):
    """An Unroller that takes Dag circuits as the input."""
    def __init__(self, dag_circuit, backend=None):
        if dag_circuit is None:
            raise UnrollerError('Invalid dag circuit!!')

        self.dag_circuit = dag_circuit
        self.set_backend(backend)

    def set_backend(self, backend):
        """Set the backend object.
        
        Give the same gate definitions to the backend circuit as
        the input circuit.
        """
        self.backend = backend
        for name, data in self.dag_circuit.gates.items():
            self.backend.define_gate(name, data)        

    def execute(self):
        """Interpret OPENQASM and make appropriate backend calls.
        
        This does not expand gates. So self.expand_gates() must have
        been previously called. Otherwise non-basis gates will be ignored
        by this method.
        """
        if self.backend is not None:
            self._process()
            return self.backend.get_output()
        else:
            raise UnrollerError("backend not attached")

    # TODO This method should merge with .execute(), so the output will depend
    # on the backend associated with this DagUnroller instance
    def expand_gates(self, basis=None):
        """Expand all gate nodes to the given basis.

        If basis is empty, each custom gate node is replaced by its
        implementation over U and CX. If basis contains some custom gates,
        then those custom gates are not expanded. For example, if "u3"
        is in basis, then the gate "u3" will not be expanded wherever
        it occurs.

        This method replicates the behavior of the unroller
        module without using the OpenQASM parser.
        """

        if basis is None:
            basis = self.backend.circuit.basis

        if not isinstance(self.backend, DAGBackend):
            raise UnrollerError("expand_gates only accepts a DAGBackend!!")

        # Build the Gate AST nodes for user-defined gates
        gatedefs = []
        for name, gate in self.dag_circuit.gates.items():
            children = [Id(name, 0, "")]
            if gate["n_args"] > 0:
                children.append(ExpressionList(list(
                    map(lambda x: Id(x, 0, ""),
                        gate["args"])
                )))
            children.append(IdList(list(
                map(lambda x: Id(x, 0, ""),
                    gate["bits"])
            )))
            children.append(gate["body"])
            gatedefs.append(Gate(children))
        
        # Walk through the DAG and examine each node
        builtins = ["U", "CX", "measure", "reset", "barrier"]
        simulator_builtins = ['snapshot', 'save', 'load', 'noise']
        topological_sorted_list = list(nx.topological_sort(self.dag_circuit.multi_graph))
        for node in topological_sorted_list:
            current_node = self.dag_circuit.multi_graph.node[node]
            if current_node["type"] == "op" and \
               current_node["op"].name not in builtins + basis + simulator_builtins and \
               not self.dag_circuit.gates[current_node["op"].name]["opaque"]:
                subcircuit, wires = self._build_subcircuit(gatedefs,
                                                           basis,
                                                           current_node["op"],
                                                           current_node["condition"])
                self.dag_circuit.substitute_circuit_one(node, subcircuit, wires)
        return self.dag_circuit

    def _build_subcircuit(self, gatedefs, basis, gate, gate_condition):
        """Build DAGCircuit for a given user-defined gate node.

        gatedefs = dictionary of Gate AST nodes for user-defined gates
        basis = basis gates used by unroller
        gate = the gate to be expanded/unrolled
        gate_condition = None or tuple (string, int)

        Returns (subcircuit, wires) where subcircuit is the DAGCircuit
        corresponding to the user-defined gate node expanded to target_basis
        and wires is the list of input wires to the subcircuit in order
        corresponding to the gate's arguments.
        """

        children = [Id(gate.name, 0, "")]
        if gate.param:
            children.append(
                ExpressionList(list(map(Real, gate.param)))
            )
        new_wires = [("q", j) for j in range(len(gate.qargs))]
        children.append(
            PrimaryList(
                list(map(lambda x: IndexedId(
                    [Id(x[0], 0, ""), Int(x[1])]
                ), new_wires))
            )
        )
        gate_node = CustomUnitary(children)
        id_int = [Id("q", 0, ""), Int(len(gate.qargs))]
        # Make a list of register declaration nodes
        reg_nodes = [
            Qreg(
                [
                    IndexedId(id_int)
                ]
            )
        ]
        # Add an If node when there is a condition present
        if gate_condition:
            gate_node = If([
                Id(gate_condition[0], 0, ""),
                Int(gate_condition[1]),
                gate_node
            ])
            new_wires += [(gate_condition[0], j)
                          for j in range(self.dag_circuit.cregs[gate_condition[0]].size)]
            reg_nodes.append(
                Creg([
                    IndexedId([
                        Id(gate_condition[0], 0, ""),
                        Int(self.dag_circuit.cregs[gate_condition[0]].size)
                    ])
                ])
            )

        # Build the whole program's AST
        sub_ast = Program(gatedefs + reg_nodes + [gate_node])
        # Interpret the AST to give a new DAGCircuit over backend basis
        sub_circuit = Unroller(sub_ast, DAGBackend(basis)).execute()
        return sub_circuit, new_wires

    def _process(self):
        """Process dag nodes, assuming that expand_gates has already been called."""
        for qreg in self.dag_circuit.qregs.values():
            self.backend.new_qreg(qreg)
        for creg in self.dag_circuit.cregs.values():
            self.backend.new_creg(creg)
        for n in nx.topological_sort(self.dag_circuit.multi_graph):
            current_node = self.dag_circuit.multi_graph.node[n]
            if current_node["type"] == "op":
                if current_node["condition"] is not None:
                    self.backend.set_condition(current_node["condition"][0],
                                               current_node["condition"][1])

                # TODO: The schema of the snapshot gate is radically
                # different to other QASM instructions. The current model
                # of extensions does not support generating custom Qobj
                # instructions (only custom QASM strings) and the default
                # instruction generator is not enough to produce a valid
                # snapshot instruction for the new Qobj format.
                #
                # This is a hack since there would be mechanisms for the
                # extensions to provide their own Qobj instructions.
                # Extensions should not be hardcoded in the DAGUnroller.
                extra_fields = None
                if current_node["op"].name == "snapshot":
                    extra_fields = {'type': 'MISSING', 'label': 'MISSING',
                                    'texparams': []}

                self.backend.start_gate(current_node["op"], extra_fields=extra_fields)
                self.backend.end_gate(current_node["op"])

                self.backend.drop_condition()

        return self.backend.get_output()
