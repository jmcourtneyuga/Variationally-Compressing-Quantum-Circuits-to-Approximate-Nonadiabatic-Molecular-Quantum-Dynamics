#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from qiskit.circuit.library.standard_gates import CXGate, RYGate
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector

def get_thetas_circuit(thetas, D2, qubits_num):
    qr = QuantumRegister(qubits_num, name="qubit")
    qc = QuantumCircuit(qr)

    qc.append(XGate(), [qr[qubits_num-1]])
    for i in range(qubits_num):
        qc.append(RYGate(thetas[i]), [qr[i]])

    for d in range(D2):
        for i in range(qubits_num-1):
            qc.append(CXGate(), [qr[i], qr[i + 1]])
        qc.barrier(qr)

        for i in range(qubits_num):
            index = qubits_num + qubits_num * d + i
            qc.append(RYGate(thetas[index]), [qr[i]])
    return qc

def get_full_variational_quantum_circuit(thetas, D2, qubits_num, input_state):
    thetas_quantum_circuit = get_thetas_circuit(thetas, D2, qubits_num)
    
    return thetas_quantum_circuit