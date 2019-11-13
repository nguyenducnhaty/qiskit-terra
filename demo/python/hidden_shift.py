# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module to solve the Hidden Shift Problem.

Given function f and g, where f is a bent function and g(x) = f(x+s), find s.

Assume oracle access to:
    - g (the shifted version of f)
    - F (the Fourier transform of f)

The oracles are represented as phase oracles, which are diagonal operators which
encode f(0), f(1), ..., f(2^n-1) into amplitude phases of computational basis
|0>, |1>, ..., |2^n-1>
"""

import math

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.quantum_initializer import DiagGate


def permutation(n_qubits, pattern=None):
    """Circuit to permute qubits.

    Args:
        n_qubits (int): circuit width
        pattern (array): permutation pattern (optional)
            If None, permute randomly.

    Returns:
        QuantumCircuit: a circuit for permuting qubits
    """
    permute = QuantumCircuit(n_qubits, name="permute")
    if pattern is not None:
        if sorted(pattern) != list(range(n_qubits)):
            raise Exception("permutation pattern must be some "
                            "ordering of 0..n_qubits-1 in a list")
        pattern = np.array(pattern)
    else:
        pattern = np.arange(n_qubits)
        np.random.shuffle(pattern)
    #print("permutation: " + str(pattern))
    for i in range(n_qubits):
        if (pattern[i] != -1) and (pattern[i] != i):
            permute.swap(i, int(pattern[i]))
            pattern[pattern[i]] = -1
    return permute


def shift(n_qubits, shift):
    """Circuit implementing bitwise xor (shift over Z_2)

    Args:
        n_qubits (int): the width of circuit
        shift (int): shift in decimal form

    Returns:
        QuantumCircuit: circuit implementing the shift
    """
    if len(bin(shift)[2:]) > n_qubits:
        raise Exception("bits in 'shift' exceed circuit width")

    circuit = QuantumCircuit(n_qubits, name="shift")

    for i in range(n_qubits):
        bit = shift&1
        shift = shift>>1
        if bit == 1:
            circuit.x(i)

    return circuit


def inner_product(n_qubits):
    """Circuit to compute inner product of 2 n-qubit registers.

    Args:
        n_qubits (int): width of top and bottom registers
            (half total circuit width)

    Returns:
        QuantumCircuit: circuit implementing the inner product
    """
    circuit = QuantumCircuit(2*n_qubits, name="inner_product")

    for i in range(n_qubits):
        circuit.cz(i, i + n_qubits)

    return circuit


def maiorana_mcfarland_oracles(n_qubits, hidden_shift, perm, boolean_function):
    """Circuit for applying a bent function from
    the Maiorana-McFarland family.

    Args:
        n_qubits (int): circuit width
        hidden_shift (int): hidden shift bitstring
        perm (array): permutation pattern of bottom register
        boolean_function (array): array of 0/1 values
            encoding a function {0, 1}^(2^n) -> {0, 1}

    Returns:
        QuantumCircuit: a quantum circuit encoding a particular
            Maiorana-McFarland bent function into amplitudes
    """
    if n_qubits % 2 != 0:
        raise Exception("bent functions only defined on even n_qubits.")

    m = int(n_qubits/2)

    top_register = QuantumRegister(m, 'x')
    bottom_register = QuantumRegister(m, 'y')

    permuter = permutation(m, perm).to_instruction()
    ip = inner_product(m).to_instruction()
    shifter = shift(n_qubits, hidden_shift).to_instruction()
    diag = DiagGate(boolean_function)

    bent_shift = QuantumCircuit(top_register, bottom_register, name="$O_g$")
    bent_shift.append(shifter, top_register[:] + bottom_register[:])
    bent_shift.barrier()
    bent_shift.append(permuter, bottom_register[:])
    bent_shift.barrier()
    bent_shift.append(ip, top_register[:] +  bottom_register[:])
    bent_shift.barrier()
    bent_shift.append(permuter.inverse(), bottom_register[:])
    bent_shift.barrier()
    bent_shift.append(diag, bottom_register[:])
    bent_shift.barrier()
    bent_shift.append(shifter.inverse(), top_register[:] + bottom_register[:])

    bent_dual = QuantumCircuit(top_register, bottom_register, name="$O_F$")
    bent_dual.append(permuter.inverse(), top_register[:])
    bent_dual.barrier()
    bent_dual.append(ip, top_register[:] +  bottom_register[:])
    bent_dual.barrier()
    bent_dual.append(diag, top_register[:])
    bent_dual.barrier()
    bent_dual.append(permuter, top_register[:])

    return bent_shift, bent_dual


def build_model_circuit(n_qubits, hidden_shift=None, perm=None, boolean_function=None):
    """Build a model circuit for solving the hidden shift problem.

    The hidden shift, and the function f, are chosen randomly.

    Args:
        n_qubits (int): width of circuit
        hidden_shift (int): hidden shift bitstring, in integer form
            if None, a random shift
        perm (array): permutation pattern used in the Maiorana-McFarland
            bent function. If None, uses random permutation.
        boolean_function (int): the boolean function used in the
            Maiorana-McFarland Bent function. If None, uses random boolean function.

    Returns:
        QuantumCircuit: a circuit that finds the hidden shift using two oracle
            invocations (the shifted bent function, f(x+s),
            and its Fourier transform F(w)).
    """
    if n_qubits % 2 != 0:
        raise Exception("n_qubits must be even for the hidden shift problem.")

    top_reg = QuantumRegister(n_qubits/2, 'x')
    bot_reg = QuantumRegister(n_qubits/2, 'y')
    hsp = QuantumCircuit(top_reg, bot_reg)

    # Step 1: Hadamard on all bits
    hsp.h(top_reg)  # top_reg + bot_reg
    hsp.h(bot_reg)
    hsp.barrier()

    #  Step 2: apply g
    #  Create a Maiorana-McFarland Bent function with a random shift pattern,
    #  a random permutation, and a random boolean function
    random_shift = (np.random.randint(1<<50)) & ((1<<n_qubits) - 1)
    random_permutation = np.arange(len(bot_reg))
    np.random.shuffle(random_permutation)
    random_boolean_function = [np.random.randint(2)*2-1 for i in range(1<<len(bot_reg))]

    s = hidden_shift or random_shift                                # hidden shift
    perm = perm or random_permutation                               # permutation pattern
    bfunc = boolean_function or random_boolean_function             # Boolean function

    bent_shift, bent_dual = maiorana_mcfarland_oracles(n_qubits, s, perm, bfunc)

    print("hidden shift: ", format(s,'#0'+str(n_qubits+2)+'b'))
    print("---------------------------")
    print("permutation pattern: ", random_permutation)
    print('boolean_function: ' + str(bfunc))

    hsp.append(bent_shift.to_instruction(), top_reg[:] + bot_reg[:])
    hsp.barrier()

    # Step 3: Hadamard on all bits again
    hsp.h(top_reg)  # top_reg + bot_reg
    hsp.h(bot_reg)
    hsp.barrier()

    # Step 4: apply F (dual of the bent function)
    hsp.append(bent_dual.to_instruction(), top_reg[:] + bot_reg[:])

    # Step 5: Hadamard on all bits one last time
    hsp.barrier()
    hsp.h(top_reg)
    hsp.h(bot_reg)

    return hsp
