# Error-Correcting Codes - Хемингов код

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import random

## 1. Какво са кодовете за корекция на грешки?

"""
Кодовете за корекция на грешки (Error-Correcting Codes) са математически схеми, 
които добавят излишна информация към данните, за да могат да:
- Откриват грешки при предаване или съхранение
- Коригират грешки без да се налага повторно предаване

Те са критично важни в:
- Телекомуникации (мобилни мрежи, сателитна връзка)
- Съхранение на данни (твърди дискове, CD/DVD/Blu-ray)
- Космически мисии (където повторното предаване е невъзможно)
- QR кодове
- RAID системи
"""

## 2. Видове кодове за корекция на грешки

"""
Основни видове:
1. **Линейни кодове**: Hamming, BCH, Reed-Solomon
2. **Конволюционни кодове**: Използват се в мобилни мрежи
3. **LDPC (Low-Density Parity-Check)**: Съвременни Wi-Fi и 5G
4. **Turbo кодове**: Сателитни комуникации

Примери от реалния свят:
- Reed-Solomon: CD/DVD дискове
- Hamming: ECC RAM памет
- BCH: SSD дискове
- Convolutional: GSM мобилни мрежи
"""

## 3. Хемингов код - История и математика

"""
Разработен от Richard Hamming в Bell Labs през 1950г.
Hamming се сблъсква с проблема на компютърните грешки всеки уикенд.

Основна идея: Използване на паритетни битове на стратегически позиции
за да се открият и коригират единични грешки.
"""

def calculate_hamming_parity_bits(data_bits: int) -> int:
    """
    Изчислява броя паритетни битове необходими за дадения брой данни
    
    Формула: 2^r >= m + r + 1
    където r = брой паритетни битове, m = брой данни битове
    """
    r = 0
    while (2**r) < (data_bits + r + 1):
        r += 1
    return r

# Демонстрация на формулата
print("Брой паритетни битове за различен брой данни:")
for data_bits in [4, 7, 11, 15, 26, 57]:
    parity_bits = calculate_hamming_parity_bits(data_bits)
    total_bits = data_bits + parity_bits
    efficiency = (data_bits / total_bits) * 100
    print(f"Данни: {data_bits:2d} битове -> Паритет: {parity_bits} -> Общо: {total_bits:2d} -> Ефективност: {efficiency:.1f}%")

## 4. Hamming Distance (Хемингово разстояние)

def hamming_distance(str1: str, str2: str) -> int:
    """Изчислява Хеминговото разстояние между два низа"""
    if len(str1) != len(str2):
        raise ValueError("Низовете трябва да са с еднаква дължина")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Демонстрация
examples = [
    ("1011101", "1001001"),
    ("karolin", "kathrin"),
    ("1000000", "0000000"),
]

print("\nПримери за Хемингово разстояние:")
for s1, s2 in examples:
    dist = hamming_distance(s1, s2)
    print(f"'{s1}' vs '{s2}' -> разстояние: {dist}")

"""
Значение на Хеминговото разстояние:
- Минималното разстояние на кода определя способността му за корекция
- За откриване на d грешки: минимално разстояние >= d + 1
- За корекция на d грешки: минимално разстояние >= 2d + 1
- Хеминговият код има минимално разстояние 3 -> може да коригира 1 грешка
"""

## 5. Имплементация на Хемингов код

class HammingCode:
    """Клас за кодиране и декодиране с Хемингов код"""
    
    def __init__(self, data_bits: int):
        self.data_bits = data_bits
        self.parity_bits = calculate_hamming_parity_bits(data_bits)
        self.total_bits = data_bits + self.parity_bits
        
        # Позициите на паритетните битове (степени на 2)
        self.parity_positions = [2**i for i in range(self.parity_bits)]
        
        print(f"Хемингов код за {data_bits} битове данни:")
        print(f"- Паритетни битове: {self.parity_bits}")
        print(f"- Общо битове: {self.total_bits}")
        print(f"- Позиции на паритетните битове: {self.parity_positions}")
    
    def encode(self, data: List[int]) -> List[int]:
        """Кодира данните с Хемингов код"""
        if len(data) != self.data_bits:
            raise ValueError(f"Данните трябва да са точно {self.data_bits} битове")
        
        # Инициализираме кодовото слово
        encoded = [0] * self.total_bits
        
        # Поставяме данните (пропускаме позициите на паритетните битове)
        data_index = 0
        for i in range(1, self.total_bits + 1):
            if i not in self.parity_positions:
                encoded[i-1] = data[data_index]
                data_index += 1
        
        # Изчисляваме паритетните битове
        for i, pos in enumerate(self.parity_positions):
            parity = 0
            for j in range(1, self.total_bits + 1):
                if j & pos:  # Проверяваме дали j-тия бит съдържа pos-та степен на 2
                    parity ^= encoded[j-1]
            encoded[pos-1] = parity
        
        return encoded
    
    def decode(self, received: List[int]) -> Tuple[List[int], int]:
        """
        Декодира получените данни и връща оригиналните данни и позицията на грешката
        Връща: (декодирани_данни, позиция_на_грешка)
        """
        if len(received) != self.total_bits:
            raise ValueError(f"Получените данни трябва да са {self.total_bits} битове")
        
        # Изчисляваме синдромния вектор
        syndrome = 0
        for i, pos in enumerate(self.parity_positions):
            parity = 0
            for j in range(1, self.total_bits + 1):
                if j & pos:
                    parity ^= received[j-1]
            if parity:
                syndrome |= pos
        
        # Ако има грешка, коригираме я
        corrected = received.copy()
        if syndrome:
            corrected[syndrome-1] ^= 1  # Обръщаме грешния бит
        
        # Извличаме оригиналните данни
        data = []
        for i in range(1, self.total_bits + 1):
            if i not in self.parity_positions:
                data.append(corrected[i-1])
        
        return data, syndrome
    
    def introduce_error(self, data: List[int], position: int) -> List[int]:
        """Въвежда грешка на зададена позиция"""
        corrupted = data.copy()
        if 0 <= position < len(data):
            corrupted[position] ^= 1
        return corrupted

## 6. Демонстрация на кодирането и декодирането

# Пример с 4-битови данни
hamming = HammingCode(4)

# Тестови данни
test_data = [1, 0, 1, 1]
print(f"\nОригинални данни: {test_data}")

# Кодиране
encoded = hamming.encode(test_data)
print(f"Кодирани данни: {encoded}")

# Показваме структурата
print("\nСтруктура на кодираното слово:")
for i in range(hamming.total_bits):
    pos = i + 1
    if pos in hamming.parity_positions:
        print(f"Позиция {pos}: {encoded[i]} (паритетен бит)")
    else:
        print(f"Позиция {pos}: {encoded[i]} (данни)")

# Тестваме без грешки
decoded, error_pos = hamming.decode(encoded)
print(f"\nДекодиране без грешки:")
print(f"Декодирани данни: {decoded}")
print(f"Позиция на грешка: {error_pos} (0 = няма грешка)")

# Тестваме с грешка
print(f"\nТестване с грешки:")
for error_position in range(hamming.total_bits):
    corrupted = hamming.introduce_error(encoded, error_position)
    decoded, detected_error = hamming.decode(corrupted)
    
    success = "✓" if decoded == test_data else "✗"
    print(f"Грешка на поз. {error_position+1}: {corrupted} -> детектирана на поз. {detected_error} {success}")

## 7. Анализ на производителността

def performance_analysis():
    """Анализира производителността на различни размери Хемингови кодове"""
    
    data_sizes = [4, 8, 16, 32, 64]
    results = []
    
    for size in data_sizes:
        hamming = HammingCode(size)
        test_data = [random.randint(0, 1) for _ in range(size)]
        
        # Измерваме времето за кодиране
        start_time = time.time()
        for _ in range(1000):
            encoded = hamming.encode(test_data)
        encode_time = (time.time() - start_time) / 1000
        
        # Измерваме времето за декодиране
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
            'encode_time': encode_time * 1000,  # в микросекунди
            'decode_time': decode_time * 1000
        })
    
    return results

# Стартираме анализа
print(f"\nАнализ на производителността:")
results = performance_analysis()

print(f"{'Данни':>6} {'Общо':>6} {'Ефект.':>8} {'Overhead':>9} {'Кодир.':>8} {'Декодир.':>9}")
print(f"{'битове':>6} {'битове':>6} {'(%)':>8} {'(%)':>9} {'(μs)':>8} {'(μs)':>9}")
print("-" * 60)

for r in results:
    print(f"{r['data_bits']:>6} {r['total_bits']:>6} {r['efficiency']:>7.1f} {r['overhead']:>8.1f} "
          f"{r['encode_time']:>7.2f} {r['decode_time']:>8.2f}")

## 8. Визуализация

def visualize_hamming_efficiency():
    """Визуализира ефективността на Хеминговия код"""
    data_bits = list(range(1, 65))
    parity_bits = [calculate_hamming_parity_bits(d) for d in data_bits]
    total_bits = [d + p for d, p in zip(data_bits, parity_bits)]
    efficiency = [(d / t) * 100 for d, t in zip(data_bits, total_bits)]
    
    plt.figure(figsize=(12, 8))
    
    # График 1: Брой паритетни битове
    plt.subplot(2, 2, 1)
    plt.plot(data_bits, parity_bits, 'b-', linewidth=2)
    plt.xlabel('Брой битове данни')
    plt.ylabel('Брой паритетни битове')
    plt.title('Паритетни битове vs Данни')
    plt.grid(True, alpha=0.3)
    
    # График 2: Ефективност
    plt.subplot(2, 2, 2)
    plt.plot(data_bits, efficiency, 'r-', linewidth=2)
    plt.xlabel('Брой битове данни')
    plt.ylabel('Ефективност (%)')
    plt.title('Ефективност на Хемингов код')
    plt.grid(True, alpha=0.3)
    
    # График 3: Сравнение общо vs данни
    plt.subplot(2, 2, 3)
    plt.plot(data_bits, data_bits, 'g-', label='Битове данни', linewidth=2)
    plt.plot(data_bits, total_bits, 'b-', label='Общо битове', linewidth=2)
    plt.plot(data_bits, parity_bits, 'r-', label='Паритетни битове', linewidth=2)
    plt.xlabel('Брой битове данни')
    plt.ylabel('Брой битове')
    plt.title('Сравнение на броя битове')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 4: Overhead
    overhead = [(p / d) * 100 for d, p in zip(data_bits, parity_bits)]
    plt.subplot(2, 2, 4)
    plt.plot(data_bits, overhead, 'm-', linewidth=2)
    plt.xlabel('Брой битове данни')
    plt.ylabel('Overhead (%)')
    plt.title('Overhead на паритетните битове')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

visualize_hamming_efficiency()

## 9. Сравнение с други методи

def simple_parity_check(data: List[int]) -> List[int]:
    """Прост паритетен контрол (може само да открива грешки)"""
    parity = sum(data) % 2
    return data + [parity]

def triple_redundancy(data: List[int]) -> List[int]:
    """Тройна избыточност - повтаря данните 3 пъти"""
    return data + data + data

def compare_error_correction_methods():
    """Сравнява различни методи за корекция на грешки"""
    
    test_data = [1, 0, 1, 1]
    print(f"\nСравнение на методи за корекция на грешки:")
    print(f"Оригинални данни: {test_data} ({len(test_data)} битове)")
    
    # Хемингов код
    hamming = HammingCode(len(test_data))
    hamming_encoded = hamming.encode(test_data)
    hamming_efficiency = (len(test_data) / len(hamming_encoded)) * 100
    
    # Прост паритет
    parity_encoded = simple_parity_check(test_data)
    parity_efficiency = (len(test_data) / len(parity_encoded)) * 100
    
    # Тройна избыточност
    triple_encoded = triple_redundancy(test_data)
    triple_efficiency = (len(test_data) / len(triple_encoded)) * 100
    
    print(f"\nРезултати:")
    print(f"{'Метод':<20} {'Битове':<8} {'Ефект.':<8} {'Може да корегира'}")
    print("-" * 50)
    print(f"{'Хемингов код':<20} {len(hamming_encoded):<8} {hamming_efficiency:.1f}%    {'1 грешка'}")
    print(f"{'Прост паритет':<20} {len(parity_encoded):<8} {parity_efficiency:.1f}%    {'Само откриване'}")
    print(f"{'Тройна избыточност':<20} {len(triple_encoded):<8} {triple_efficiency:.1f}%    {'1 грешка'}")

compare_error_correction_methods()

## 10. Ограничения и подобрения

"""
Ограничения на Хеминговия код:
1. Може да коригира само ЕДНА грешка
2. Може да открива само ДВЕ грешки
3. При повече грешки може да доведе до неправилна корекция

Подобрения:
1. **Extended Hamming Code**: Добавя още един паритетен бит за по-добро откриване
2. **BCH кодове**: Могат да коригират множество грешки
3. **Reed-Solomon**: Използва се в CD/DVD, може да коригира burst грешки
4. **LDPC кодове**: Използват се в модерни комуникации (Wi-Fi 6, 5G)

Реални приложения:
- ECC RAM: Използва SECDED (Single Error Correction, Double Error Detection)
- SSD дискове: BCH или LDPC кодове
- Космически мисии: Convolutional + Reed-Solomon кодове
- QR кодове: Reed-Solomon кодове
"""

## 11. Тест с реални данни

def test_with_text():
    """Тества Хемингов код с текстови данни"""
    
    text = "HELLO"
    print(f"\nТест с текст: '{text}'")
    
    # Конвертираме текста в битове
    binary_data = []
    for char in text:
        # Всеки ASCII символ = 7 битове
        ascii_val = ord(char)
        bits = [(ascii_val >> i) & 1 for i in range(7)]
        binary_data.extend(bits)
    
    print(f"Битова репрезентация: {binary_data} ({len(binary_data)} битове)")
    
    # Използваме Хемингов код
    hamming = HammingCode(len(binary_data))
    encoded = hamming.encode(binary_data)
    
    print(f"Кодирано: {encoded} ({len(encoded)} битове)")
    print(f"Ефективност: {(len(binary_data)/len(encoded)*100):.1f}%")
    
    # Симулираме грешка
    error_pos = len(encoded) // 2
    corrupted = hamming.introduce_error(encoded, error_pos)
    decoded, detected_error = hamming.decode(corrupted)
    
    # Конвертираме обратно в текст
    recovered_text = ""
    for i in range(0, len(decoded), 7):
        if i + 6 < len(decoded):
            char_bits = decoded[i:i+7]
            ascii_val = sum(bit * (2**j) for j, bit in enumerate(char_bits))
            recovered_text += chr(ascii_val)
    
    print(f"Възстановен текст: '{recovered_text}'")
    print(f"Успех: {'✓' if recovered_text == text else '✗'}")

test_with_text()

print(f"\n" + "="*60)
print(f"ЗАКЛЮЧЕНИЕ:")
print(f"="*60)
print(f"""
Хеминговият код е елегантно решение за корекция на единични грешки:

ПРЕДИМСТВА:
+ Ефективен за малки блокове данни
+ Прост за имплементация
+ Гарантирана корекция на 1 грешка
+ Математически елегантен

НЕДОСТАТЪЦИ:
- Ограничен до корекция на 1 грешка
- Ефективността намалява при малки блокове
- Не може да се справя с burst грешки

СЪВРЕМЕННИ АЛТЕРНАТИВИ:
- BCH кодове: Multiple error correction
- Reed-Solomon: Burst error correction  
- LDPC кодове: Near-optimal performance
- Turbo кодове: Excellent performance

Хеминговият код остава важен като основа за разбиране на 
принципите на корекцията на грешки и се използва в 
специализирани приложения като ECC RAM памет.
""")
