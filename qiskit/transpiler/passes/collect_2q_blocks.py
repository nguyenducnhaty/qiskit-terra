# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


import networkx as nx
from qiskit.transpiler.basepasses import TransformationPass


class Collect2qBlocks(TransformationPass):
    """
    Expand a gate in a circuit using its decomposition rules.
    """

    def __init__(self):
        """pass to collect sequences of uninterrupted gates acting on 2 qubits."""
        super.__init__()

    def run(self, dag):
         """Collect blocks of adjacent gates acting on a pair of "cx" qubits.
 
         The blocks contain "op" nodes in topological sort order
         such that all gates in a block act on the same pair of
         qubits and are adjacent in the circuit. The blocks are built
         by examining predecessors and successors of "cx" gates in
         the circuit. Gates with names "u1, u2, u3, cx, id" will be included.
 
         Return a list of tuples of "op" node labels.
         """
         good_names = ["cx", "u1", "u2", "u3", "id"]
         block_list = []
         ts = list(nx.topological_sort(dag.multi_graph))
         nodes_seen = dict(zip(ts, [False] * len(ts)))
         for node in ts:
             nd = dag.multi_graph.node[node]
             group = []
             # Explore predecessors and successors of cx gates
            if nd["name"] == "cx" and len(nd["cargs"]) == 0 \
               and nd["condition"] is None and not nodes_seen[node]:
                these_qubits = sorted(nd["qargs"])
                print("// looking at node %d on %s" % (node, str(these_qubits)))
                #print("// looking at node %d on %s" % (node, str(these_qubits)))
                # Explore predecessors of the "cx" node
                pred = list(dag.multi_graph.predecessors(node))
                explore = True
                while explore:
                    pred_next = []
                    print("//   pred = %s, types = %s" % (str(pred), str(list(map(lambda x: dag.multi_graph.node[x]["name"], pred)))))
                    #print("//   pred = %s, types = %s" % (str(pred), str(list(map(lambda x: dag.multi_graph.node[x]["name"], pred)))))
                    # If there is one predecessor, add it if its on the right qubits
                    if len(pred) == 1 and not nodes_seen[pred[0]]:
                        pnd = dag.multi_graph.node[pred[0]]
                         if pnd["name"] in good_names:
                             if (pnd["name"] == "cx" and sorted(pnd["qargs"]) == these_qubits) or \
                                 pnd["name"] != "cx":
                                 group.append(pred[0])
                                 nodes_seen[pred[0]] = True
                                 pred_next.extend(dag.multi_graph.predecessors(pred[0]))
                     # If there are two, then we consider cases
                     elif len(pred) == 2:
                         # First, check if there is a relationship
                         if pred[0] in dag.multi_graph.predecessors(pred[1]):
                             sorted_pred = [pred[1]]   # was [pred[1], pred[0]]
                        elif pred[1] in dag.multi_graph.predecessors(pred[0]):
                            sorted_pred = [pred[0]]   # was [pred[0], pred[1]]
                        else:
                            sorted_pred = pred
                        if dag.multi_graph.node[sorted_pred[0]]["name"] == "cx" and \
                            # We need to avoid accidentally adding a cx on these_qubits
                            # since these must have a dependency through the other predecessor
                            # in this case
                            if dag.multi_graph.node[pred[0]]["name"] == "cx" and \
                               sorted(dag.multi_graph.node[pred[0]]["qargs"]) == these_qubits:
                               sorted_pred = [pred[1]]
                            elif dag.multi_graph.node[pred[1]]["name"] == "cx" and \
                                sorted(dag.multi_graph.node[pred[1]]["qargs"]) == these_qubits:
                                sorted_pred = [pred[0]]
                            else:
                                sorted_pred = pred
                        if len(sorted_pred) == 2 and \
                           dag.multi_graph.node[sorted_pred[0]]["name"] == "cx" and \
                           dag.multi_graph.node[sorted_pred[1]]["name"] == "cx":
                            break  # stop immediately if we hit a pair of cx
                        # Examine each predecessor
                         for p in sorted_pred:
                             pnd = dag.multi_graph.node[p]
                             if pnd["name"] not in good_names:
                                 continue
                             # If a predecessor is a single qubit gate, add it
                             if pnd["name"] != "cx":
                                 if not nodes_seen[p]:
                                     group.append(p)
                                     nodes_seen[p] = True
                                     pred_next.extend(dag.multi_graph.predecessors(p))
                             else:
                                 # If cx, check qubits
                                 pred_qubits = sorted(pnd["qargs"])
                                 if pred_qubits == these_qubits:
                                     # add if on same qubits
                                     if not nodes_seen[p]:
                                         group.append(p)
                                         nodes_seen[p] = True
                                         pred_next.extend(dag.multi_graph.predecessors(p))
                                 else:
                                     # remove qubit from consideration if not
                                     these_qubits = list(set(these_qubits) -
                                                         set(pred_qubits))
                     # Update predecessors
                     # Stop if there aren't any more
                     pred = list(set(pred_next))
                     if len(pred) == 0:
                         explore = False
                 # Reverse the predecessor list and append the "cx" node
                 group.reverse()
                 group.append(node)
                 nodes_seen[node] = True
                 # Reset these_qubits
                 these_qubits = sorted(nd["qargs"])
                 # Explore successors of the "cx" node
                 succ = list(dag.multi_graph.successors(node))
                explore = True
                while explore:
                    succ_next = []
                    print("//   succ = %s, types = %s" % (str(succ), str(list(map(lambda x: dag.multi_graph.node[x]["name"], succ)))))
                    #print("//   succ = %s, types = %s" % (str(succ), str(list(map(lambda x: dag.multi_graph.node[x]["name"], succ)))))
                    # If there is one successor, add it if its on the right qubits
                    if len(succ) == 1 and not nodes_seen[succ[0]]:
                        snd = dag.multi_graph.node[succ[0]]
@@ -1402,8 +1413,19 @@ def collect_cx_blocks(dag):
                        elif succ[1] in dag.multi_graph.successors(succ[0]):
                            sorted_succ = [succ[0]]  # was [succ[0], succ[1]]
                        else:
                            sorted_succ = succ
                        if dag.multi_graph.node[sorted_succ[0]]["name"] == "cx" and \
                            # We need to avoid accidentally adding a cx on these_qubits
                            # since these must have a dependency through the other successor
                            # in this case
                            if dag.multi_graph.node[succ[0]]["name"] == "cx" and \
                               sorted(dag.multi_graph.node[succ[0]]["qargs"]) == these_qubits:
                               sorted_succ = [succ[1]]
                            elif dag.multi_graph.node[succ[1]]["name"] == "cx" and \
                                 sorted(dag.multi_graph.node[succ[1]]["qargs"]) == these_qubits:
                                 sorted_succ = [succ[0]]
                            else:
                                sorted_succ = succ
                        if len(sorted_succ) == 2 and \
                           dag.multi_graph.node[sorted_succ[0]]["name"] == "cx" and \
                           dag.multi_graph.node[sorted_succ[1]]["name"] == "cx":
                            break  # stop immediately if we hit a pair of cx
                        # Examine each successor

        return dag
