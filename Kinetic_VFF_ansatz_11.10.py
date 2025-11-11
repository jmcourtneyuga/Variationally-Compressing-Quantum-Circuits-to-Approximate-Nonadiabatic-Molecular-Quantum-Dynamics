#!/usr/bin/env python
# coding: utf-8

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.standard_gates import RZGate, CXGate
from qiskit.quantum_info import Operator
import numpy as np
from functools import lru_cache
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def ZZ(qc, qr, qubit_1, qubit_2, theta):
    qc.append(CXGate(), [qr[qubit_1], qr[qubit_2]])
    qc.append(RZGate(theta), [qr[qubit_2]])
    qc.append(CXGate(), [qr[qubit_1], qr[qubit_2]])

def _build_W_operations(thetas_W, n_qubits, num_layers_eigenvector):
    idx = 0
    all_ops = []
        
    # Apply eigenvector layers
    for layer in range(num_layers_eigenvector):
        layer_ops = []

        # 2. Two-qubit gates on even-odd pairs sublayer
        for qubit in range(0, n_qubits-1, 2):
            layer_ops.append(("ZZ", qubit, qubit + 1, thetas_W[idx]))
            idx += 1
        
        # 3. Two-qubit gates on odd-even pairs sublayer
        for qubit in range(1, n_qubits-1, 2):
            layer_ops.append(("ZZ", qubit, qubit + 1, thetas_W[idx]))    
            idx += 1
            
        all_ops.append(layer_ops)
    return all_ops, idx

def build_W_circuit(qc, qr, thetas_W, n_qubits, num_layers_eigenvector):
    idx = 0
    eigenvector_ops = []

    # Apply eigenvector layers
    for layer in range(num_layers_eigenvector):
        layer_ops = []

        # 2. Two-qubit gates on even-odd pairs sublayer
        for qubit in range(0, n_qubits-1, 2):
            ZZ(qc, qr, qubit, qubit + 1, thetas_W[idx])
            layer_ops.append(("ZZ", qubit, qubit + 1, thetas_W[idx]))
            idx += 1
        
        # 3. Two-qubit gates on odd-even pairs sublayer
        for qubit in range(1, n_qubits-1, 2):
            ZZ(qc, qr, qubit, qubit + 1, thetas_W[idx])
            layer_ops.append(("ZZ", qubit, qubit + 1, thetas_W[idx]))    
            idx += 1
        
        eigenvector_ops.append(layer_ops)
    return qc, eigenvector_ops, idx

def build_D_circuit(qc, qr, thetas_D, n_qubits, num_layers_diagonal):
    idx = 0
    for layer in range(num_layers_diagonal):
        # 1. RZ gates on each qubit
        for qubit in range(n_qubits):
            qubit_param = thetas_D[idx]
            idx += 1
            qc.append(RZGate(qubit_param), [qr[qubit]])

        # 2. ZZ gates on even-odd pairs
        for qubit in range(0, n_qubits-1, 2):
            qubit_param = thetas_D[idx]
            idx += 1
            ZZ(qc, qr, qubit, qubit+1, qubit_param)
            
        # 3. ZZ gates on odd-even pairs
        for qubit in range(1, n_qubits-1, 2):
            qubit_param = thetas_D[idx]
            idx += 1
            ZZ(qc, qr, qubit, qubit+1, qubit_param)

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 0, 2, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 3, 5, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 1, 3, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 4, 6, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 2, 4, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 5, 7, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 0, 3, qubit_param) 
        
        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 4, 7, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 1, 4, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 2, 5, qubit_param) 

        qubit_param = thetas_D[idx]
        idx +=1
        ZZ(qc, qr, 3, 6, qubit_param) 

    return qc

def build_W_dagger_circuit(qc, qr, eigenvector_ops, n_qubits):
    # Process operations in reverse order (for dagger)
    for layer_ops in reversed(eigenvector_ops):
        for op in reversed(layer_ops):
            if op[0] == "RZ":
                _, qubit, theta = op
                qc.append(RZGate(-theta), [qr[qubit]]) 
    
            elif op[0] == "ZZ":
                _, q1, q2, theta = op
                ZZ(qc, qr, q1, q2, -theta)

    return qc

# Cache the parameter count calculations to avoid redundant computations
@lru_cache(maxsize=16)
def calculate_parameter_counts(n_qubits, num_layers_eigenvector, num_layers_diagonal):
    w_params_per_layer = n_qubits//2 + (n_qubits - 1)//2
    total_w_params = num_layers_eigenvector * w_params_per_layer

    rz_params = n_qubits
    zz_even_params = n_qubits // 2
    zz_odd_params = (n_qubits - 1) // 2

    d_params_per_layer = (rz_params + zz_even_params + zz_odd_params) + 11

    total_d_params = num_layers_diagonal * d_params_per_layer

    return total_w_params, total_d_params

def get_full_vff_quantum_circuit(thetas, num_layers_eigenvector, num_layers_diagonal, n_qubits):
    total_w_params, total_d_params = calculate_parameter_counts(n_qubits, num_layers_eigenvector, num_layers_diagonal)
    
    # Slice thetas into W and D parts
    thetas = np.array(thetas, dtype=float)
    thetas_W_forward = thetas[:total_w_params].copy()  # For first W
    thetas_W_dagger = thetas[:total_w_params].copy()   # For W†
    thetas_D = thetas[total_w_params:total_w_params + total_d_params].copy()
    
    # Build the circuit
    qr = QuantumRegister(n_qubits, name='qubit')
    qc = QuantumCircuit(qr)
    
    qc, eigenvector_ops_forward, w_used = build_W_circuit(qc, qr, thetas_W_forward, n_qubits, num_layers_eigenvector)
    build_D_circuit(qc, qr, thetas_D, n_qubits, num_layers_diagonal)
    build_W_dagger_circuit(qc, qr, eigenvector_ops_forward, n_qubits)

    return qc