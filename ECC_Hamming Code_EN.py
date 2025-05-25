# Problem 8. Error-Correcting Codes - Hamming Code

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import random

## 1. What are Error-Correcting Codes?

"""
Error-Correcting Codes (ECCs) are mathematical schemes that add redundant 
information to data to enable:
- Detection of errors during transmission or storage
- Correction of errors without requiring retransmission

They are critically important in:
- Telecommunications (mobile networks, satellite communication)
- Data storage (hard drives, CD/DVD/Blu-ray)
- Space missions (where retransmission is impossible)
- QR codes
- RAID systems
"""

## 2. Types of Error-Correcting Codes

"""
Main types:
1. **Linear codes**: Hamming, BCH, Reed-Solomon
2. **Convolutional codes**: Used in mobile networks
3. **LDPC (Low-Density Parity-Check)**: Modern Wi-Fi and 5G
4. **Turbo codes**: Satellite communications

Real-world examples:
- Reed-Solomon: CD/DVD discs
- Hamming: ECC RAM memory
- BCH: SSD drives
- Convolutional: GSM mobile networks
"""

## 3. Hamming Code - History and Mathematics

"""
Developed by Richard Hamming at Bell Labs in 1950.
Hamming encountered the problem of computer errors every weekend.

Core idea: Use parity bits at strategic positions to detect 
and correct single errors.
"""

def calculate_hamming_parity_bits(data_bits: int) -> int:
    """
    Calculates the number of parity bits needed for given data bits
    
    Formula: 2^r >= m + r + 1
    where r = number of parity bits, m = number of data bits
    """
    r = 0
    while (2**r) < (data_bits + r + 1):
        r += 1
    return r

# Demonstration of the formula
print("Number of parity bits for different data sizes:")
for data_bits in [4, 7, 11, 15, 26, 57]:
    parity_bits = calculate_hamming_parity_bits(data_bits)
    total_bits = data_bits + parity_bits
    efficiency = (data_bits / total_bits) * 100
    print(f"Data: {data_bits:2d} bits -> Parity: {parity_bits} -> Total: {total_bits:2d} -> Efficiency: {efficiency:.1f}%")

## 4. Hamming Distance

def hamming_distance(str1: str, str2: str) -> int:
    """Calculates the Hamming distance between two strings"""
    if len(str1) != len(str2):
        raise ValueError("Strings must have equal length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Demonstration
examples = [
    ("1011101", "1001001"),
    ("karolin", "kathrin"),
    ("1000000", "0000000"),
]

print("\nHamming distance examples:")
for s1, s2 in examples:
    dist = hamming_distance(s1, s2)
    print(f"'{s1}' vs '{s2}' -> distance: {dist}")

"""
Significance of Hamming distance:
- Minimum distance of the code determines its correction capability
- To detect d errors: minimum distance >= d + 1
- To correct d errors: minimum distance >= 2d + 1
- Hamming code has minimum distance 3 -> can correct 1 error
"""

## 5. Hamming Code Implementation

class HammingCode:
    """Class for Hamming code encoding and decoding"""
    
    def __init__(self, data_bits: int):
        self.data_bits = data_bits
        self.parity_bits = calculate_hamming_parity_bits(data_bits)
        self.total_bits = data_bits + self.parity_bits
        
        # Parity bit positions (powers of 2)
        self.parity_positions = [2**i for i in range(self.parity_bits)]
        
        print(f"Hamming code for {data_bits} data bits:")
        print(f"- Parity bits: {self.parity_bits}")
        print(f"- Total bits: {self.total_bits}")
        print(f"- Parity positions: {self.parity_positions}")
    
    def encode(self, data: List[int]) -> List[int]:
        """Encodes data using Hamming code"""
        if len(data) != self.data_bits:
            raise ValueError(f"Data must be exactly {self.data_bits} bits")
        
        # Initialize codeword
        encoded = [0] * self.total_bits
        
        # Place data bits (skip parity positions)
        data_index = 0
        for i in range(1, self.total_bits + 1):
            if i not in self.parity_positions:
                encoded[i-1] = data[data_index]
                data_index += 1
        
        # Calculate parity bits
        for i, pos in enumerate(self.parity_positions):
            parity = 0
            for j in range(1, self.total_bits + 1):
                if j & pos:  # Check if j-th bit contains pos-th power of 2
                    parity ^= encoded[j-1]
            encoded[pos-1] = parity
        
        return encoded
    
    def decode(self, received: List[int]) -> Tuple[List[int], int]:
        """
        Decodes received data and returns original data and error position
        Returns: (decoded_data, error_position)
        """
        if len(received) != self.total_bits:
            raise ValueError(f"Received data must be {self.total_bits} bits")
        
        # Calculate syndrome vector
        syndrome = 0
        for i, pos in enumerate(self.parity_positions):
            parity = 0
            for j in range(1, self.total_bits + 1):
                if j & pos:
                    parity ^= received[j-1]
            if parity:
                syndrome |= pos
        
        # If there's an error, correct it
        corrected = received.copy()
        if syndrome:
            corrected[syndrome-1] ^= 1  # Flip the erroneous bit
        
        # Extract original data
        data = []
        for i in range(1, self.total_bits + 1):
            if i not in self.parity_positions:
                data.append(corrected[i-1])
        
        return data, syndrome
    
    def introduce_error(self, data: List[int], position: int) -> List[int]:
        """Introduces an error at specified position"""
        corrupted = data.copy()
        if 0 <= position < len(data):
            corrupted[position] ^= 1
        return corrupted

## 6. Encoding and Decoding Demonstration

# Example with 4-bit data
hamming = HammingCode(4)

# Test data
test_data = [1, 0, 1, 1]
print(f"\nOriginal data: {test_data}")

# Encoding
encoded = hamming.encode(test_data)
print(f"Encoded data: {encoded}")

# Show structure
print("\nStructure of encoded word:")
for i in range(hamming.total_bits):
    pos = i + 1
    if pos in hamming.parity_positions:
        print(f"Position {pos}: {encoded[i]} (parity bit)")
    else:
        print(f"Position {pos}: {encoded[i]} (data bit)")

# Test without errors
decoded, error_pos = hamming.decode(encoded)
print(f"\nDecoding without errors:")
print(f"Decoded data: {decoded}")
print(f"Error position: {error_pos} (0 = no error)")

# Test with errors
print(f"\nTesting with errors:")
for error_position in range(hamming.total_bits):
    corrupted = hamming.introduce_error(encoded, error_position)
    decoded, detected_error = hamming.decode(corrupted)
    
    success = "✓" if decoded == test_data else "✗"
    print(f"Error at pos. {error_position+1}: {corrupted} -> detected at pos. {detected_error} {success}")

## 7. Performance Analysis

def performance_analysis():
    """Analyzes performance of different Hamming code sizes"""
    
    data_sizes = [4, 8, 16, 32, 64]
    results = []
    
    for size in data_sizes:
        hamming = HammingCode(size)
        test_data = [random.randint(0, 1) for _ in range(size)]
        
        # Measure encoding time
        start_time = time.time()
        for _ in range(1000):
            encoded = hamming.encode(test_data)
        encode_time = (time.time() - start_time) / 1000
        
        # Measure decoding time
        start_time = time.time()
        for _ in range(1000):
            decoded, _ = hamming.decode(encoded)
        decode_time = (time.time() - start_time) / 1000
        
        efficiency = (size / hamming.total_bits) * 100
        overhead = ((hamming.total_bits - size) / size) * 100
        
        results.append({
            'data_bits': size,
            'total_bits': hamming.total_bits,
            'efficiency': efficiency,
            'overhead': overhead,
            'encode_time': encode_time * 1000,  # in microseconds
            'decode_time': decode_time * 1000
        })
    
    return results

# Run analysis
print(f"\nPerformance Analysis:")
results = performance_analysis()

print(f"{'Data':>6} {'Total':>6} {'Effic.':>8} {'Overhead':>9} {'Encode':>8} {'Decode':>9}")
print(f"{'bits':>6} {'bits':>6} {'(%)':>8} {'(%)':>9} {'(μs)':>8} {'(μs)':>9}")
print("-" * 60)

for r in results:
    print(f"{r['data_bits']:>6} {r['total_bits']:>6} {r['efficiency']:>7.1f} {r['overhead']:>8.1f} "
          f"{r['encode_time']:>7.2f} {r['decode_time']:>8.2f}")

## 8. Visualization

def visualize_hamming_efficiency():
    """Visualizes Hamming code efficiency"""
    data_bits = list(range(1, 65))
    parity_bits = [calculate_hamming_parity_bits(d) for d in data_bits]
    total_bits = [d + p for d, p in zip(data_bits, parity_bits)]
    efficiency = [(d / t) * 100 for d, t in zip(data_bits, total_bits)]
    
    plt.figure(figsize=(12, 8))
    
    # Graph 1: Number of parity bits
    plt.subplot(2, 2, 1)
    plt.plot(data_bits, parity_bits, 'b-', linewidth=2)
    plt.xlabel('Number of data bits')
    plt.ylabel('Number of parity bits')
    plt.title('Parity bits vs Data bits')
    plt.grid(True, alpha=0.3)
    
    # Graph 2: Efficiency
    plt.subplot(2, 2, 2)
    plt.plot(data_bits, efficiency, 'r-', linewidth=2)
    plt.xlabel('Number of data bits')
    plt.ylabel('Efficiency (%)')
    plt.title('Hamming Code Efficiency')
    plt.grid(True, alpha=0.3)
    
    # Graph 3: Comparison total vs data
    plt.subplot(2, 2, 3)
    plt.plot(data_bits, data_bits, 'g-', label='Data bits', linewidth=2)
    plt.plot(data_bits, total_bits, 'b-', label='Total bits', linewidth=2)
    plt.plot(data_bits, parity_bits, 'r-', label='Parity bits', linewidth=2)
    plt.xlabel('Number of data bits')
    plt.ylabel('Number of bits')
    plt.title('Bit count comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graph 4: Overhead
    overhead = [(p / d) * 100 for d, p in zip(data_bits, parity_bits)]
    plt.subplot(2, 2, 4)
    plt.plot(data_bits, overhead, 'm-', linewidth=2)
    plt.xlabel('Number of data bits')
    plt.ylabel('Overhead (%)')
    plt.title('Parity bit overhead')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_hamming_efficiency()

## 9. Comparison with Other Methods

def simple_parity_check(data: List[int]) -> List[int]:
    """Simple parity check (can only detect errors)"""
    parity = sum(data) % 2
    return data + [parity]

def triple_redundancy(data: List[int]) -> List[int]:
    """Triple redundancy - repeats data 3 times"""
    return data + data + data

def compare_error_correction_methods():
    """Compares different error correction methods"""
    
    test_data = [1, 0, 1, 1]
    print(f"\nComparison of error correction methods:")
    print(f"Original data: {test_data} ({len(test_data)} bits)")
    
    # Hamming code
    hamming = HammingCode(len(test_data))
    hamming_encoded = hamming.encode(test_data)
    hamming_efficiency = (len(test_data) / len(hamming_encoded)) * 100
    
    # Simple parity
    parity_encoded = simple_parity_check(test_data)
    parity_efficiency = (len(test_data) / len(parity_encoded)) * 100
    
    # Triple redundancy
    triple_encoded = triple_redundancy(test_data)
    triple_efficiency = (len(test_data) / len(triple_encoded)) * 100
    
    print(f"\nResults:")
    print(f"{'Method':<20} {'Bits':<8} {'Effic.':<8} {'Can correct'}")
    print("-" * 50)
    print(f"{'Hamming code':<20} {len(hamming_encoded):<8} {hamming_efficiency:.1f}%    {'1 error'}")
    print(f"{'Simple parity':<20} {len(parity_encoded):<8} {parity_efficiency:.1f}%    {'Detection only'}")
    print(f"{'Triple redundancy':<20} {len(triple_encoded):<8} {triple_efficiency:.1f}%    {'1 error'}")

compare_error_correction_methods()

## 10. Limitations and Improvements

"""
Limitations of Hamming code:
1. Can only correct ONE error
2. Can only detect TWO errors
3. With more errors may lead to incorrect correction

Improvements:
1. **Extended Hamming Code**: Adds another parity bit for better detection
2. **BCH codes**: Can correct multiple errors
3. **Reed-Solomon**: Used in CD/DVD, can correct burst errors
4. **LDPC codes**: Used in modern communications (Wi-Fi 6, 5G)

Real applications:
- ECC RAM: Uses SECDED (Single Error Correction, Double Error Detection)
- SSD drives: BCH or LDPC codes
- Space missions: Convolutional + Reed-Solomon codes
- QR codes: Reed-Solomon codes
"""

## 11. Testing with Real Data

def test_with_text():
    """Tests Hamming code with text data"""
    
    text = "HELLO"
    print(f"\nText test: '{text}'")
    
    # Convert text to bits
    binary_data = []
    for char in text:
        # Each ASCII character = 7 bits
        ascii_val = ord(char)
        bits = [(ascii_val >> i) & 1 for i in range(7)]
        binary_data.extend(bits)
    
    print(f"Binary representation: {binary_data} ({len(binary_data)} bits)")
    
    # Use Hamming code
    hamming = HammingCode(len(binary_data))
    encoded = hamming.encode(binary_data)
    
    print(f"Encoded: {encoded} ({len(encoded)} bits)")
    print(f"Efficiency: {(len(binary_data)/len(encoded)*100):.1f}%")
    
    # Simulate error
    error_pos = len(encoded) // 2
    corrupted = hamming.introduce_error(encoded, error_pos)
    decoded, detected_error = hamming.decode(corrupted)
    
    # Convert back to text
    recovered_text = ""
    for i in range(0, len(decoded), 7):
        if i + 6 < len(decoded):
            char_bits = decoded[i:i+7]
            ascii_val = sum(bit * (2**j) for j, bit in enumerate(char_bits))
            recovered_text += chr(ascii_val)
    
    print(f"Recovered text: '{recovered_text}'")
    print(f"Success: {'✓' if recovered_text == text else '✗'}")

test_with_text()

## 12. Comparison with Python Libraries

def compare_with_library():
    """Compare our implementation with existing libraries"""
    print(f"\nComparison with standard implementations:")
    print(f"Note: For production use, consider libraries like:")
    print(f"- pyecc: Python Error Correction Codes")
    print(f"- commpy: Digital Communication with Python")
    print(f"- galois: Galois field arithmetic")
    print(f"")
    print(f"Our implementation advantages:")
    print(f"+ Educational clarity")
    print(f"+ Full control over parameters")
    print(f"+ Easy to modify and extend")
    print(f"")
    print(f"Library advantages:")
    print(f"+ Optimized performance")
    print(f"+ Extended functionality")
    print(f"+ Thoroughly tested")
    print(f"+ Support for advanced codes")

compare_with_library()

## 13. Advanced Error Correction Discussion

def discuss_advanced_codes():
    """Discusses more advanced error correction codes"""
    
    print(f"\nAdvanced Error Correction Codes:")
    print(f"="*50)
    
    codes = [
        {
            "name": "Reed-Solomon",
            "capability": "Multiple error correction, burst errors",
            "applications": "CD/DVD, QR codes, space communication",
            "complexity": "Medium",
            "efficiency": "Good for burst errors"
        },
        {
            "name": "BCH (Bose-Chaudhuri-Hocquenghem)",
            "capability": "Multiple error correction",
            "applications": "SSD drives, digital TV",
            "complexity": "Medium-High",
            "efficiency": "Good for random errors"
        },
        {
            "name": "LDPC (Low-Density Parity-Check)",
            "capability": "Near Shannon limit performance",
            "applications": "Wi-Fi 6, 5G, satellite",
            "complexity": "High",
            "efficiency": "Excellent"
        },
        {
            "name": "Turbo Codes",
            "capability": "Near Shannon limit performance",
            "applications": "3G/4G mobile, deep space",
            "complexity": "High",
            "efficiency": "Excellent"
        }
    ]
    
    for code in codes:
        print(f"\n{code['name']}:")
        print(f"  Capability: {code['capability']}")
        print(f"  Applications: {code['applications']}")
        print(f"  Complexity: {code['complexity']}")
        print(f"  Efficiency: {code['efficiency']}")

discuss_advanced_codes()

print(f"\n" + "="*60)
print(f"CONCLUSION:")
print(f"="*60)
print(f"""
Hamming code is an elegant solution for single error correction:

ADVANTAGES:
+ Efficient for small data blocks
+ Simple to implement
+ Guaranteed correction of 1 error
+ Mathematically elegant

DISADVANTAGES:
- Limited to correcting 1 error
- Efficiency decreases with small blocks
- Cannot handle burst errors

MODERN ALTERNATIVES:
- BCH codes: Multiple error correction
- Reed-Solomon: Burst error correction  
- LDPC codes: Near-optimal performance
- Turbo codes: Excellent performance

Hamming code remains important as a foundation for understanding
error correction principles and is used in specialized applications
like ECC RAM memory.
""")
