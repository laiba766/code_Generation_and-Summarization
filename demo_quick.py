"""
Quick Demo - Code Generation and Summarization (No Model Download)
Uses template/rule-based approaches for instant results
"""

import sys
sys.path.append('src')

from code_generator import CodeGenerator
from code_summarizer import CodeSummarizer

print("="*80)
print("CODE GENERATION & SUMMARIZATION DEMO (TEMPLATE-BASED)")
print("="*80)

# Initialize with None to force template-based approach
print("\n⏳ Initializing...")
generator = CodeGenerator()
generator.model = None  # Force template mode
summarizer = CodeSummarizer()
summarizer.model = None  # Force rule-based mode
print("✅ Ready!")

# ============================================================================
# DEMO 1: CODE GENERATION
# ============================================================================
print("\n" + "="*80)
print("DEMO 1: GENERATE CODE FROM ENGLISH DESCRIPTION")
print("="*80)

descriptions = [
    ("Function to calculate the factorial of a number", "python"),
    ("Function to check if a string is a palindrome", "python"),
    ("Class to manage a list of students", "java"),
]

for desc, lang in descriptions:
    print(f"\n📝 Description: {desc}")
    print(f"🔤 Language: {lang.upper()}")
    print("-"*80)
    code = generator.generate_code(desc, lang)
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
        "language": "python",
        "code": """
def bubble_sort(arr):
    '''Sort an array using bubble sort algorithm'''
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
        "language": "python",
        "code": """
def binary_search(arr, target):
    '''Search for target in sorted array using binary search'''
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
        "name": "File Reader",
        "language": "python",
        "code": """
import pandas as pd

def read_csv_file(filename):
    '''Read a CSV file and return as DataFrame'''
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
"""
    }
]

for example in code_examples:
    print(f"\n💻 Code Example: {example['name']} ({example['language'].upper()})")
    print("-"*80)
    print(example['code'])
    print("-"*80)
    
    summary = summarizer.summarize_code(example['code'], example['language'])
    print(f"\n📝 Summary:\n  {summary}")
    
    print("\n📊 Detailed Analysis:")
    details = summarizer.summarize_with_details(example['code'], example['language'])
    for key, value in details.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print("="*80)

# ============================================================================
# DEMO 3: REAL PROJECT CODE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("DEMO 3: ANALYZE REAL PROJECT CODE")
print("="*80)

# Read and summarize one of our project files
import os

project_file = "src/data_collector.py"
if os.path.exists(project_file):
    print(f"\n📄 Analyzing: {project_file}")
    print("-"*80)
    
    with open(project_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Get first 100 lines only
    lines = code.split('\n')[:100]
    code_sample = '\n'.join(lines)
    
    print("\n📊 Analysis:")
    details = summarizer.summarize_with_details(code_sample, "python")
    for key, value in details.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print("="*80)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ DEMO COMPLETED!")
print("="*80)
print("\n📌 KEY FEATURES:")
print("  1. ✅ Generate code from English descriptions")
print("  2. ✅ Summarize existing code into natural language")
print("  3. ✅ Support multiple programming languages (Python, Java, JavaScript)")
print("  4. ✅ Extract key operations and dependencies")
print("  5. ✅ Assess code complexity")

print("\n💡 TIP: For better results, you can:")
print("  • Use the full AI models (requires downloading)")
print("  • Run: python code_interface.py --interactive")
print("  • Or use the programmatic API in your own scripts")

print("\n📚 See CODE_GENERATION_GUIDE.md for detailed instructions")
print("="*80)
