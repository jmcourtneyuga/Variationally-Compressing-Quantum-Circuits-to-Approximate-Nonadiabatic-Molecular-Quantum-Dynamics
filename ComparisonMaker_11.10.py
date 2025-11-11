#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer  

from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library import XGate, CCXGate
from qiskit.quantum_info import Statevector
import re

def generate_less_than_logic(number, n_bits):
    binary = bin(number)[2:].zfill(n_bits)
    if number == 0:
    one_positions = []
    for i in range(n_bits):
        if binary[i] == '1':
            one_positions.append(i)
    if not one_positions:
        return "Always false (no unsigned bit string can be less than 0)"

    msb_index = one_positions[0]
    prefix = []
    for i in range(msb_index):
        bit_pos = n_bits - i - 1
        prefix.append(f"b_{bit_pos} = 0")
    
    result = build_less_than_condition(binary, msb_index, n_bits)
    if prefix:
        return " AND ".join(prefix) + " AND " + result
    else:
        return result

def build_less_than_condition(binary, start_index, n_bits):
    if start_index >= n_bits:
        return ""

    bit_pos = n_bits - start_index - 1
    bit_val = binary[start_index]

    if bit_val == '0':
        next_condition = build_less_than_condition(binary, start_index + 1, n_bits)
        if next_condition:
            return f"b_{bit_pos} = 0 AND {next_condition}"
        else:
            return f"b_{bit_pos} = 0"
    subsequent_ones = []
    for i in range(start_index + 1, n_bits):
        if binary[i] == '1':
            subsequent_ones.append(i)

    if not subsequent_ones:
        return f"b_{bit_pos} = 0"
    
    next_one_index = subsequent_ones[0]

    middle_condition = []
    for i in range(start_index + 1, next_one_index):
        middle_bit_pos = n_bits - i - 1
        middle_condition.append(f"b_{middle_bit_pos} = 0")

    first_option = f"b_{bit_pos} = 0"

    second_option = ""
    if middle_condition:
        second_option = " AND ".join(middle_condition)
    
    rest_condition = build_less_than_condition(binary, next_one_index, n_bits)
    
    if second_option and rest_condition:
        return f"{first_option} OR ({second_option} AND {rest_condition})"
    elif second_option:
        return f"{first_option} OR ({second_option})"
    elif rest_condition:
        return f"{first_option} OR ({rest_condition})"
    else:
        return first_option

def extract_all_zero_bits_from_logic(logic):
    if "Always" in logic:
        return []
    
    # Find all matches of the pattern b_X = 0
    pattern = r'b_(\d+) = 0'
    matches = re.findall(pattern, logic)
    
    # Convert all matches to integers and create a set to remove duplicates
    required_zero_bits = set(int(match) for match in matches)
    
    return sorted(list(required_zero_bits))

def get_less_than_logic_explanation(number, n_bits):
    binary_repr = bin(number)[2:].zfill(n_bits)
    logic = generate_less_than_logic(number, n_bits)
    
    explanation = f"Number: {number} (binary: {binary_repr})\n"
    explanation += "-" * 60 + "\n"
    explanation += f"For a {n_bits}-bit string to be LESS THAN {number}:\n"
    explanation += logic + "\n"
    
    return explanation

def get_less_than_operations(number, n_bits, register_name="position_register"):
    binary_repr = bin(number)[2:].zfill(n_bits)
    logic = generate_less_than_logic(number, n_bits)
    explanation = get_less_than_logic_explanation(number, n_bits)
    
    # Extract all bits that need to be 0, including those in OR conditions
    if "Always" in logic:
        zero_bits = []
        operations = []
        qiskit_operations = []
    else:
        zero_bits = extract_all_zero_bits_from_logic(logic)
        operations = [(XGate(), [f"{register_name}[{bit_pos}]"]) for bit_pos in zero_bits]
        qiskit_operations = [f"qc.x(qr[{bit_pos}])" for bit_pos in zero_bits]
    
    return {
        "number": number,
        "binary": binary_repr,
        "logic": logic,
        "x_gate_positions": zero_bits,
        "operations": operations,
        "qiskit_operations": qiskit_operations,
        "explanation": explanation
    }

def create_less_than_qiskit_circuit(number, n_bits, verbose=False):
    result = get_less_than_operations(number, n_bits)
    binary_repr = result["binary"]
    logic = result["logic"]
    zero_bits = result["x_gate_positions"]
    
    # Print original logic explanation if verbose
    if verbose:
        print(result["explanation"])
    
    # Create Qiskit circuit
    if "Always" in logic:
        if verbose:
            print("Quantum Circuit:")
            print("No circuit needed - result is constant")
        return None
    
    # Initialize quantum register
    qr = QuantumRegister(n_bits, 'q')
    cr = ClassicalRegister(1, 'c')  # 1 bit for the result
    qc = QuantumCircuit(qr, cr)
    for bit_pos in zero_bits:
        qc.x(qr[bit_pos])
    
    if verbose:
        print(qc.draw(output='text'))
        
        print(f"\nThe circuit applies X gates to qubits that must be 0 in the logical condition.")
        print(f"These qubits are: {zero_bits}")
        print(f"After X gates, if all relevant combinations of these qubits are 1, the input is less than {number}")
        
        # Print the operations
        print("\nQiskit operations needed:")
        for op in result["qiskit_operations"]:
            print(op)
        
        print("\nGate operations format:")
        for op in result["operations"]:
            print(op)
    
    return qc

def build_greater_than_condition(binary, start_index, n_bits):
    if start_index >= n_bits:
        return ""

    bit_pos = n_bits - start_index - 1
    bit_val = binary[start_index]

    if bit_val == '1':
        next_condition = build_greater_than_condition(binary, start_index + 1, n_bits)
        if next_condition:
            return f"b_{bit_pos} = 1 AND {next_condition}"
        else:
            return f"b_{bit_pos} = 1"

    subsequent_zeros = []
    for i in range(start_index + 1, n_bits):
        if binary[i] == '0':
            subsequent_zeros.append(i)

    if not subsequent_zeros:
        remaining_bits = []
        for i in range(start_index + 1, n_bits):
            bit_pos_i = n_bits - i - 1
            if binary[i] == '0': 
                remaining_bits.append(f"b_{bit_pos_i} = 1")
        
        if remaining_bits:
            return f"b_{bit_pos} = 1 OR ({' AND '.join(remaining_bits)})"
        else:
            return f"b_{bit_pos} = 1"

    next_zero_index = subsequent_zeros[0]

    middle_condition = []
    for i in range(start_index + 1, next_zero_index):
        middle_bit_pos = n_bits - i - 1
        if binary[i] == '1':
            middle_condition.append(f"b_{middle_bit_pos} = 1")

    first_option = f"b_{bit_pos} = 1"

    second_option = ""
    if middle_condition:
        second_option = " AND ".join(middle_condition)

    rest_condition = build_greater_than_condition(binary, next_zero_index, n_bits)
    if second_option and rest_condition:
        return f"{first_option} OR ({second_option} AND {rest_condition})"
    elif second_option:
        return f"{first_option} OR ({second_option})"
    elif rest_condition:
        return f"{first_option} OR ({rest_condition})"
    else:
        return first_option

def generate_greater_than_logic(number, n_bits):
    binary = bin(number)[2:].zfill(n_bits)

    if int(binary, 2) == (2**n_bits - 1):
        return f"Always false (cannot be greater than maximum {n_bits}-bit value)"

    zero_positions = []
    for i in range(n_bits):
        if binary[i] == '0':
            zero_positions.append(i)

    if not zero_positions:
        return f"Always false (cannot be greater than maximum {n_bits}-bit value)"

    msb_zero_index = zero_positions[0]

    prefix = []
    for i in range(msb_zero_index):
        bit_pos = n_bits - i - 1
        bit_val = binary[i]
        prefix.append(f"b_{bit_pos} = {bit_val}")

    result = build_greater_than_condition(binary, msb_zero_index, n_bits)

    if prefix:
        return " AND ".join(prefix) + " AND " + result
    else:
        return result