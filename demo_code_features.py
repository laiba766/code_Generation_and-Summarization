"""
Demo Script for Code Generation and Summarization
Shows examples of both features working
"""

import sys
sys.path.append('src')

from code_generator import CodeGenerator
from code_summarizer import CodeSummarizer

print("="*80)
print("CODE GENERATION & SUMMARIZATION DEMO")
print("="*80)

# Initialize
print("\n⏳ Initializing models...")
generator = CodeGenerator()
summarizer = CodeSummarizer()
print("✅ Models ready!")

# ============================================================================
# DEMO 1: CODE GENERATION
# ============================================================================
print("\n" + "="*80)
print("DEMO 1: GENERATE CODE FROM ENGLISH DESCRIPTION")
print("="*80)

descriptions = [
    "Function to calculate the factorial of a number",
    "Function to check if a string is a palindrome",
    "Function to find the maximum value in a list"
]

for i, desc in enumerate(descriptions, 1):
    print(f"\n📝 Description {i}: {desc}")
    print("-"*80)
    code = generator.generate_code(desc, "python")
    print("✨ Generated Code:")
    print(code)
    print("-"*80)

# ============================================================================
# DEMO 2: CODE SUMMARIZATION
# ============================================================================
print("\n" + "="*80)
print("DEMO 2: SUMMARIZE EXISTING CODE")
print("="*80)

code_examples = [
    {
        "name": "Bubble Sort",
        "code": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    },
    {
        "name": "Binary Search",
        "code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    },
    {
        "name": "Fibonacci",
        "code": """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    }
]

for example in code_examples:
    print(f"\n💻 Code Example: {example['name']}")
    print("-"*80)
    print(example['code'])
    print("-"*80)
    
    summary = summarizer.summarize_code(example['code'], "python")
    print(f"📝 Summary: {summary}")
    
    print("\n📊 Detailed Analysis:")
    details = summarizer.summarize_with_details(example['code'], "python")
    for key, value in details.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print("-"*80)

# ============================================================================
# DEMO 3: MULTIPLE VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("DEMO 3: GENERATE MULTIPLE CODE VARIATIONS")
print("="*80)

desc = "Function to reverse a string"
print(f"\n📝 Description: {desc}")
print("-"*80)

variations = generator.generate_multiple_variations(desc, "python", num_variations=2)
for i, code in enumerate(variations, 1):
    print(f"\n✨ Variation {i}:")
    print(code)
    print("-"*80)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ DEMO COMPLETED!")
print("="*80)
print("\nYou can now:")
print("  1. Generate code from English descriptions")
print("  2. Summarize existing code into natural language")
print("  3. Get multiple variations of generated code")
print("  4. Get detailed analysis of code complexity and operations")
print("\nRun interactive mode with:")
print("  python code_interface.py --interactive")
print("="*80)
