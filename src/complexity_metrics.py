"""
Complexity Metrics Module
Implements McCabe Cyclomatic Complexity and Halstead Metrics
"""

import ast
import re
import math
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McCabeComplexity:
    """Calculate McCabe's Cyclomatic Complexity"""

    def __init__(self):
        self.complexity = 0

    def calculate_python(self, code: str) -> int:
        """Calculate cyclomatic complexity for Python code"""
        try:
            tree = ast.parse(code)
            return self._visit_ast(tree)
        except SyntaxError:
            logger.warning("Syntax error in Python code")
            return 1

    def _visit_ast(self, node) -> int:
        """Visit AST nodes and count decision points"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Conditional statements
            if isinstance(child, ast.If):
                complexity += 1

            # Loops
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                complexity += 1

            # Exception handling
            if isinstance(child, (ast.ExceptHandler,)):
                complexity += 1

            # Boolean operators
            if isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

            # Comprehensions
            if isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                complexity += 1

            # Function calls with lambda
            if isinstance(child, ast.Lambda):
                complexity += 1

        return complexity

    def calculate_generic(self, code: str) -> int:
        """Calculate cyclomatic complexity using heuristic for any language"""
        complexity = 1  # Base complexity

        # Count decision points
        decision_keywords = [
            r'\bif\b', r'\belse\s+if\b', r'\belif\b',
            r'\bfor\b', r'\bwhile\b',
            r'\bcase\b', r'\bcatch\b',
            r'\&\&', r'\|\|',
            r'\?', r'\:'
        ]

        for keyword in decision_keywords:
            matches = re.findall(keyword, code)
            complexity += len(matches)

        return complexity

    def calculate(self, code: str, language: str) -> int:
        """Calculate McCabe complexity based on language"""
        if language == 'Python':
            return self.calculate_python(code)
        else:
            return self.calculate_generic(code)


class HalsteadMetrics:
    """Calculate Halstead Complexity Metrics"""

    def __init__(self):
        self.operators = set()
        self.operands = set()
        self.total_operators = 0
        self.total_operands = 0

    def extract_python_tokens(self, code: str) -> Dict[str, Any]:
        """Extract operators and operands from Python code"""
        operators = []
        operands = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Arithmetic operators
                if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                                     ast.Pow, ast.FloorDiv)):
                    operators.append(type(node).__name__)

                # Bitwise operators
                if isinstance(node, (ast.LShift, ast.RShift, ast.BitOr,
                                     ast.BitXor, ast.BitAnd)):
                    operators.append(type(node).__name__)

                # Logical operators
                if isinstance(node, (ast.And, ast.Or, ast.Not)):
                    operators.append(type(node).__name__)

                # Comparison operators
                if isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                                     ast.Gt, ast.GtE, ast.Is, ast.IsNot,
                                     ast.In, ast.NotIn)):
                    operators.append(type(node).__name__)

                # Unary operators
                if isinstance(node, (ast.UAdd, ast.USub, ast.Invert)):
                    operators.append(type(node).__name__)

                # Assignment
                if isinstance(node, ast.Assign):
                    operators.append('=')

                # Operands - variables and literals
                if isinstance(node, ast.Name):
                    operands.append(node.id)

                if isinstance(node, ast.Constant):
                    operands.append(str(node.value))

                # Function/method names
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        operands.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        operands.append(node.func.attr)

        except SyntaxError:
            logger.warning("Syntax error while extracting Python tokens")

        return {
            'operators': operators,
            'operands': operands
        }

    def extract_generic_tokens(self, code: str) -> Dict[str, Any]:
        """Extract operators and operands using regex for generic languages"""
        operators = []
        operands = []

        # Operator patterns
        operator_patterns = [
            r'\+', r'-', r'\*', r'/', r'%', r'\^',
            r'==', r'!=', r'<=', r'>=', r'<', r'>',
            r'&&', r'\|\|', r'!',
            r'=', r'\+=', r'-=', r'\*=', r'/=',
            r'&', r'\|', r'~', r'<<', r'>>',
            r'\?', r':'
        ]

        for pattern in operator_patterns:
            matches = re.findall(pattern, code)
            operators.extend([pattern] * len(matches))

        # Operand patterns (simplified)
        # Variables and identifiers
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = re.findall(var_pattern, code)

        # Filter out keywords
        keywords = {
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break',
            'continue', 'return', 'function', 'var', 'let', 'const', 'class',
            'public', 'private', 'protected', 'static', 'void', 'int', 'float',
            'double', 'string', 'boolean', 'true', 'false', 'null', 'undefined',
            'fn', 'mut', 'impl', 'trait', 'struct', 'enum', 'match'
        }

        operands = [v for v in variables if v not in keywords]

        # Numeric literals
        num_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(num_pattern, code)
        operands.extend(numbers)

        # String literals
        string_pattern = r'["\'].*?["\']'
        strings = re.findall(string_pattern, code)
        operands.extend(strings)

        return {
            'operators': operators,
            'operands': operands
        }

    def calculate_metrics(self, operators: List[str], operands: List[str]) -> Dict[str, float]:
        """Calculate Halstead metrics from operators and operands"""
        n1 = len(set(operators))  # Distinct operators
        n2 = len(set(operands))    # Distinct operands
        N1 = len(operators)        # Total operators
        N2 = len(operands)         # Total operands

        # Avoid division by zero
        if n1 == 0 or n2 == 0:
            return {
                'halstead_n1': n1,
                'halstead_n2': n2,
                'halstead_N1': N1,
                'halstead_N2': N2,
                'halstead_vocabulary': n1 + n2,
                'halstead_length': N1 + N2,
                'halstead_volume': 0,
                'halstead_difficulty': 0,
                'halstead_effort': 0
            }

        # Calculate metrics
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2.0) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume

        return {
            'halstead_n1': n1,
            'halstead_n2': n2,
            'halstead_N1': N1,
            'halstead_N2': N2,
            'halstead_vocabulary': vocabulary,
            'halstead_length': length,
            'halstead_volume': volume,
            'halstead_difficulty': difficulty,
            'halstead_effort': effort
        }

    def calculate(self, code: str, language: str) -> Dict[str, float]:
        """Calculate Halstead metrics based on language"""
        if language == 'Python':
            tokens = self.extract_python_tokens(code)
        else:
            tokens = self.extract_generic_tokens(code)

        return self.calculate_metrics(tokens['operators'], tokens['operands'])


class ComplexityCalculator:
    """Main class to calculate all complexity metrics"""

    def __init__(self):
        self.mccabe = McCabeComplexity()
        self.halstead = HalsteadMetrics()

    def calculate_all(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate all complexity metrics"""
        metrics = {}

        # McCabe complexity
        metrics['cc_mccabe'] = self.mccabe.calculate(code, language)

        # Halstead metrics
        halstead_metrics = self.halstead.calculate(code, language)
        metrics.update(halstead_metrics)

        # Additional basic metrics
        lines = code.split('\n')
        metrics['loc'] = len([l for l in lines if l.strip()])
        metrics['num_blank_lines'] = len([l for l in lines if not l.strip()])
        metrics['num_comment_lines'] = len([l for l in lines if l.strip().startswith(('#', '//'))])

        return metrics


if __name__ == "__main__":
    # Example usage
    calculator = ComplexityCalculator()

    sample_code = """
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
"""

    metrics = calculator.calculate_all(sample_code, 'Python')
    print("Complexity Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
