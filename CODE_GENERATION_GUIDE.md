# Code Generation and Summarization

This document explains how to use the code generation and summarization features.

## Features

1. **Code Generation**: Write descriptions in English and generate code
2. **Code Summarization**: Convert existing code into natural language summaries

## Quick Start

### Interactive Mode (Recommended)

```bash
# Run in interactive mode (both generation and summarization)
python code_interface.py --interactive

# Generate code only
python code_interface.py --mode generate --interactive

# Summarize code only
python code_interface.py --mode summarize --interactive
```

### Examples

#### 1. Generate Code from English Description

```bash
python code_interface.py --mode generate --language python --interactive
```

Then type descriptions like:
- "Function to calculate the factorial of a number"
- "Class to manage a list of students with add, remove, and search methods"
- "Function to read a CSV file and return a pandas DataFrame"

#### 2. Summarize Existing Code

```bash
python code_interface.py --mode summarize --language python --interactive
```

Then paste your code and type `END` when finished.

#### 3. Process Files

**Generate code from a description file:**
```bash
echo "Function to sort a list of numbers in ascending order" > description.txt
python code_interface.py --mode generate --file description.txt --language python
```

**Summarize a code file:**
```bash
python code_interface.py --mode summarize --file src/data_collector.py --language python
```

## Supported Languages

- Python
- Java  
- JavaScript

## Usage Guide

### Code Generation

1. **Single Generation**:
   - Enter your description in plain English
   - The tool generates code based on your description
   - Code is displayed immediately

2. **Multiple Variations**:
   - After generating code, you can request multiple variations
   - Each variation uses different sampling parameters for diversity

3. **Example-Based Generation**:
   ```python
   from src.code_generator import CodeGenerator
   
   generator = CodeGenerator()
   examples = [
       ("5", "120"),  # 5! = 120
       ("3", "6"),    # 3! = 6
   ]
   code = generator.generate_from_examples(
       "Calculate factorial", 
       examples, 
       "python"
   )
   ```

### Code Summarization

1. **Basic Summary**:
   - Paste your code
   - Get a concise natural language description

2. **Detailed Analysis**:
   - Overview summary
   - Complexity assessment
   - Key operations identified
   - Dependencies extracted

3. **Batch Processing**:
   ```python
   from src.code_summarizer import CodeSummarizer
   
   summarizer = CodeSummarizer()
   code_snippets = [code1, code2, code3]
   summaries = summarizer.batch_summarize(code_snippets, "python")
   ```

## Advanced Usage

### Programmatic API

**Code Generation:**
```python
from src.code_generator import CodeGenerator

generator = CodeGenerator()

# Basic generation
code = generator.generate_code(
    "Function to find the maximum value in a list",
    language="python"
)

# Multiple variations
variations = generator.generate_multiple_variations(
    "Function to reverse a string",
    language="python",
    num_variations=3
)
```

**Code Summarization:**
```python
from src.code_summarizer import CodeSummarizer

summarizer = CodeSummarizer()

code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

# Basic summary
summary = summarizer.summarize_code(code, "python")
print(summary)

# Detailed summary
details = summarizer.summarize_with_details(code, "python")
for key, value in details.items():
    print(f"{key}: {value}")
```

## Models Used

- **Code Generation**: Salesforce CodeGen (350M parameters)
  - Fallback: Template-based generation
- **Code Summarization**: Salesforce CodeT5
  - Fallback: Rule-based analysis

## Tips

1. **For Better Generation**:
   - Be specific in your descriptions
   - Mention input/output types if important
   - Include key algorithms or approaches

2. **For Better Summarization**:
   - Include docstrings and comments in your code
   - Use meaningful variable and function names
   - Keep functions focused on single tasks

## Examples

### Example 1: Generate a Binary Search Function

**Input:**
```
Function to perform binary search on a sorted list and return the index
```

**Output:**
```python
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
```

### Example 2: Summarize Code

**Input:**
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

**Output:**
```
Defines function: quicksort. Implements recursive sorting algorithm using pivot-based 
partitioning. Contains conditional logic and list comprehensions. Returns sorted array.
```

## Troubleshooting

1. **Model Loading Issues**: If models fail to load, the tool automatically falls back to template/rule-based approaches
2. **Memory Issues**: Use smaller code snippets or reduce batch sizes
3. **Language Support**: Currently optimized for Python, with basic support for Java and JavaScript
