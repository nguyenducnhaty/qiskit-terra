from qiskit import *
q = QuantumRegister(3)  	            # anonymous quantum register
bell_circ = QuantumCircuit(q)	            # anonymous quantum circuit
bell_circ.h(q[0])                           # make a bell state
bell_circ.cx(q[0], q[1])

from qiskit.tools.visualization import *
plot_circuit(bell_circ)

cnot_circ = QuantumCircuit(q)               # define a new circuit on the same register
cnot_circ.cx(q[0], q[2])
ghz_circ = bell_circ + cnot_circ            # combine it with the bell circuit to create a ghz state
plot_circuit(ghz_circ)

c = ClassicalRegister(3)                    # define a new circuit on different registers
measure_circ = QuantumCircuit(q, c)
measure_circ.measure(q, c)
ghz_measure_circ = ghz_circ + measure_circ  # combine to create ghz circuit with final measurements
plot_circuit(ghz_measure_circ, scale=.3)

print(available_backends())                 # QISKit is installed with a few default simulators

result = execute(ghz_measure_circ, 'local_qasm_simulator').result()
counts = result.get_counts()
plot_histogram(counts)

result = execute(ghz_circ, 'local_statevector_simulator').result()
result.get_statevector()

import qiskit.extensions.simulator
bell_circ.snapshot('1')
ghz_circ = bell_circ + cnot_circ
result = execute(ghz_circ, 'local_statevector_simulator').result()
result.get_snapshot()

import Qconfig
register(Qconfig.APItoken)                  # grant access to remote devices and simulators
print(available_backends())

available_backends({'local': False, 'simulator': True})

job = execute(ghz_measure_circ, 'ibmqx4', shots=1000)
import time
while not job.done:                         # check job status every 10 seconds
    time.sleep(10)
job.result()                                # retrieve the result when complete

result = job.result()
counts = result.get_counts()
plot_histogram(counts)
