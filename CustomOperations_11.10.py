#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer  

from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library import XGate, CCXGate, CRXGate
from qiskit.quantum_info import Statevector
import re

def ccrx(circuit, theta, ctrl1, ctrl2, target):
    circuit.cx(ctrl2, target)
    circuit.append(CRXGate(theta/2), [ctrl1, target])
    circuit.cx(ctrl2, target)
    circuit.append(CRXGate(-theta/2), [ctrl1, target])
    circuit.cx(ctrl1, ctrl2)
    circuit.append(CRXGate(theta/2), [ctrl2, target])
    circuit.cx(ctrl1, ctrl2)
    circuit.append(CRXGate(-theta/2), [ctrl2, target])

def ccp(circuit, theta, ctrl1, ctrl2, target):
    circuit.cx(ctrl2, target)
    circuit.cp(theta/2, ctrl1, target)
    circuit.cx(ctrl2, target)
    circuit.cp(-theta/2, ctrl1, target)
    circuit.cx(ctrl1, ctrl2)
    circuit.cp(theta/2, ctrl2, target)
    circuit.cx(ctrl1, ctrl2)
    circuit.cp(-theta/2, ctrl2, target)

def ccrx_inverse(circuit, theta, ctrl1, ctrl2, target):
    """Constructs the inverse of a doubly-controlled RX (CCRX) gate."""
    circuit.append(CRXGate(theta/2), [ctrl2, target])
    circuit.cx(ctrl1, ctrl2)
    circuit.append(CRXGate(-theta/2), [ctrl2, target])
    circuit.cx(ctrl1, ctrl2)
    circuit.append(CRXGate(theta/2), [ctrl1, target])
    circuit.cx(ctrl2, target)
    circuit.append(CRXGate(-theta/2), [ctrl1, target])
    circuit.cx(ctrl2, target)

def cqft(qc, register, n_qubits):
    for i in range(n_qubits // 2):
        qc.swap(register[i], register[n_qubits - 1 - i])

    for i in range(n_qubits):
        qc.h(register[i])
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, register[j], register[i])

    qc.x(register[n_qubits-1]) 

def ciqft(qc, register, n_qubits):
    qc.x(register[n_qubits-1])  

    for i in reversed(range(n_qubits)):
        for j in reversed(range(i + 1, n_qubits)):
            angle = -np.pi / (2 ** (j - i))
            qc.cp(angle, register[j], register[i])
        qc.h(register[i])

    for i in range(n_qubits // 2):
        qc.swap(register[i], register[n_qubits - 1 - i])

def qft(qc, register, n_qubits):
    for i in range(n_qubits // 2):
        qc.swap(register[i], register[n_qubits - 1 - i])

    for i in range(n_qubits):
        qc.h(register[i])
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, register[j], register[i])

def iqft(qc, register, n_qubits):
    for i in reversed(range(n_qubits)):
        for j in reversed(range(i + 1, n_qubits)):
            angle = -np.pi / (2 ** (j - i))
            qc.cp(angle, register[j], register[i])
        qc.h(register[i])

    for i in range(n_qubits // 2):
        qc.swap(register[i], register[n_qubits - 1 - i])

def X(target):
    operations = [
        (XGate(), [target])
    ]
    return operations

def IF(ctrl, target):
    operations = [
        (CXGate(), [ctrl, target])
    ]
    return operations

def AND(ctrl1, ctrl2, target):
    operations = [
        (CCXGate(), [ctrl1, ctrl2, target])
    ]
    return operations

def invAND(ctrl1, ctrl2, target):
    operations = AND(ctrl1, ctrl2, target)
    return operations

def OR(ctrl1, ctrl2, target):
    operations = [
        (XGate(), [ctrl1]),
        (XGate(), [ctrl2]),
        (XGate(), [target]),
        (CCXGate(), [ctrl1, ctrl2, target]),
        (XGate(), [ctrl1]),
        (XGate(), [ctrl2])
    ]
    return operations

def invOR(ctrl1, ctrl2, target):
    operations = list(reversed(OR(ctrl1, ctrl2, target)))
    return operations