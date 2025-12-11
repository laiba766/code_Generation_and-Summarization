"""
AST Parser Module
Parses code into Abstract Syntax Trees and extracts structural features
"""

import ast
import re
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASTAnalyzer:
    """Analyzes Abstract Syntax Trees to extract structural metrics"""

    def __init__(self):
        self.ast_tree = None
        self.language = None

    def parse_python(self, code: str) -> Optional[ast.AST]:
        """Parse Python code into AST"""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Python syntax error: {e}")
            return None

    def calculate_ast_depth(self, node, current_depth=0) -> int:
        """Calculate maximum depth of AST"""
        if not hasattr(node, '_fields') or not node._fields:
            return current_depth

        max_child_depth = current_depth
        for field_name in node._fields:
            field_value = getattr(node, field_name, None)

            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        depth = self.calculate_ast_depth(item, current_depth + 1)
                        max_child_depth = max(max_child_depth, depth)
            elif isinstance(field_value, ast.AST):
                depth = self.calculate_ast_depth(field_value, current_depth + 1)
                max_child_depth = max(max_child_depth, depth)

        return max_child_depth

    def count_nodes(self, node) -> int:
        """Count total number of nodes in AST"""
        count = 1
        for child in ast.walk(node):
            if child != node:
                count += 1
        return count

    def count_leaf_nodes(self, node) -> int:
        """Count leaf nodes (nodes with no children)"""
        if not hasattr(node, '_fields') or not node._fields:
            return 1

        has_ast_children = False
        leaf_count = 0

        for field_name in node._fields:
            field_value = getattr(node, field_name, None)

            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        has_ast_children = True
                        leaf_count += self.count_leaf_nodes(item)
            elif isinstance(field_value, ast.AST):
                has_ast_children = True
                leaf_count += self.count_leaf_nodes(field_value)

        return leaf_count if has_ast_children else 1

    def calculate_branching_factor(self, node, depths=None) -> float:
        """Calculate average branching factor"""
        if depths is None:
            depths = []

        children_count = 0
        for field_name in node._fields:
            field_value = getattr(node, field_name, None)

            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, ast.AST):
                        children_count += 1
                        self.calculate_branching_factor(item, depths)
            elif isinstance(field_value, ast.AST):
                children_count += 1
                self.calculate_branching_factor(field_value, depths)

        if children_count > 0:
            depths.append(children_count)

        return sum(depths) / len(depths) if depths else 0

    def count_node_types(self, node) -> int:
        """Count distinct AST node types"""
        node_types = set()
        for child in ast.walk(node):
            node_types.add(type(child).__name__)
        return len(node_types)

    def count_statements_expressions(self, tree) -> Dict[str, int]:
        """Count different types of statements and expressions"""
        counts = {
            'num_statements': 0,
            'num_expressions': 0,
            'num_if': 0,
            'num_for': 0,
            'num_while': 0,
            'num_try_catch': 0,
            'num_function_calls': 0,
            'num_return': 0
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                counts['num_statements'] += 1

            if isinstance(node, ast.expr):
                counts['num_expressions'] += 1

            if isinstance(node, ast.If):
                counts['num_if'] += 1

            if isinstance(node, (ast.For, ast.AsyncFor)):
                counts['num_for'] += 1

            if isinstance(node, ast.While):
                counts['num_while'] += 1

            if isinstance(node, (ast.Try, ast.TryExcept, ast.TryFinally)):
                counts['num_try_catch'] += 1

            if isinstance(node, ast.Call):
                counts['num_function_calls'] += 1

            if isinstance(node, ast.Return):
                counts['num_return'] += 1

        return counts

    def extract_operators_operands(self, tree) -> Dict[str, Any]:
        """Extract operators and operands for Halstead metrics"""
        operators = []
        operands = []

        for node in ast.walk(tree):
            # Operators
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                                 ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
                                 ast.BitXor, ast.BitAnd, ast.FloorDiv)):
                operators.append(type(node).__name__)

            if isinstance(node, (ast.And, ast.Or, ast.Not)):
                operators.append(type(node).__name__)

            if isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
                                 ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                operators.append(type(node).__name__)

            # Operands
            if isinstance(node, ast.Name):
                operands.append(node.id)

            if isinstance(node, (ast.Num, ast.Str, ast.Bytes)):
                operands.append(str(node))

            if isinstance(node, ast.Constant):
                operands.append(str(node.value))

        return {
            'operators': operators,
            'operands': operands,
            'n1': len(set(operators)),  # distinct operators
            'n2': len(set(operands)),    # distinct operands
            'N1': len(operators),         # total operators
            'N2': len(operands)           # total operands
        }

    def analyze_python_ast(self, code: str) -> Dict[str, Any]:
        """Analyze Python AST and extract all metrics"""
        tree = self.parse_python(code)
        if tree is None:
            return None

        features = {}

        # Basic AST metrics
        features['ast_node_count'] = self.count_nodes(tree)
        features['ast_depth'] = self.calculate_ast_depth(tree)
        features['ast_leaf_count'] = self.count_leaf_nodes(tree)
        features['ast_branching_factor_avg'] = self.calculate_branching_factor(tree)
        features['ast_distinct_node_types'] = self.count_node_types(tree)

        # Statement and expression counts
        stmt_counts = self.count_statements_expressions(tree)
        features.update(stmt_counts)

        # Operators and operands
        op_features = self.extract_operators_operands(tree)
        features.update(op_features)

        # Control flow features
        features['has_loops'] = 1 if (features['num_for'] > 0 or features['num_while'] > 0) else 0
        features['has_exception_handling'] = 1 if features['num_try_catch'] > 0 else 0

        return features

    def analyze_generic(self, code: str, language: str) -> Dict[str, Any]:
        """Generic analysis using regex patterns for non-Python languages"""
        features = {}

        lines = code.split('\n')
        features['loc'] = len([l for l in lines if l.strip() and not l.strip().startswith(('//', '#', '/*'))])

        # Count control structures
        features['num_if'] = len(re.findall(r'\bif\s*\(', code))
        features['num_for'] = len(re.findall(r'\bfor\s*\(', code))
        features['num_while'] = len(re.findall(r'\bwhile\s*\(', code))
        features['num_try_catch'] = len(re.findall(r'\btry\s*\{', code))
        features['num_function_calls'] = len(re.findall(r'\w+\s*\(', code))
        features['num_return'] = len(re.findall(r'\breturn\b', code))

        # Approximate AST metrics
        features['ast_node_count'] = features['loc'] * 3  # rough estimate
        features['ast_depth'] = min(20, features['loc'] // 3)  # rough estimate
        features['has_loops'] = 1 if (features['num_for'] > 0 or features['num_while'] > 0) else 0
        features['has_exception_handling'] = 1 if features['num_try_catch'] > 0 else 0

        return features

    def analyze(self, code: str, language: str) -> Dict[str, Any]:
        """Main analysis method"""
        if language == 'Python':
            return self.analyze_python_ast(code)
        else:
            return self.analyze_generic(code, language)


class ConcurrencyFeatureExtractor:
    """Extracts concurrency-related features from code"""

    def __init__(self):
        self.patterns = {
            'Python': {
                'async': r'\basync\s+def\b',
                'await': r'\bawait\b',
                'asyncio': r'\basyncio\.',
                'threading': r'\bthreading\.',
                'multiprocessing': r'\bmultiprocessing\.'
            },
            'Java': {
                'thread': r'\bnew\s+Thread\b',
                'executor': r'\bExecutorService\b',
                'synchronized': r'\bsynchronized\b',
                'lock': r'\bLock\b',
                'future': r'\bFuture\b'
            },
            'JavaScript': {
                'async': r'\basync\s+function\b|\basync\s*\(',
                'await': r'\bawait\b',
                'promise': r'\bnew\s+Promise\b|\bPromise\.',
                'then': r'\.then\(',
                'setTimeout': r'\bsetTimeout\b',
                'setInterval': r'\bsetInterval\b'
            },
            'Rust': {
                'spawn': r'\bspawn\s*\(',
                'mutex': r'\bMutex\b',
                'arc': r'\bArc\b',
                'channel': r'\bchannel\b',
                'async': r'\basync\s+fn\b'
            }
        }

    def extract(self, code: str, language: str) -> Dict[str, Any]:
        """Extract concurrency features"""
        features = {}

        if language not in self.patterns:
            return {
                'uses_concurrency': 0,
                'concurrency_pattern': 'none',
                'count_async': 0,
                'count_await': 0,
                'count_thread': 0,
                'count_mutex': 0,
                'count_promise': 0
            }

        lang_patterns = self.patterns[language]

        features['count_async'] = len(re.findall(lang_patterns.get('async', r'$^'), code))
        features['count_await'] = len(re.findall(lang_patterns.get('await', r'$^'), code))
        features['count_thread'] = len(re.findall(lang_patterns.get('thread', r'$^'), code)) + \
                                    len(re.findall(lang_patterns.get('threading', r'$^'), code))
        features['count_mutex'] = len(re.findall(lang_patterns.get('mutex', r'$^'), code)) + \
                                   len(re.findall(lang_patterns.get('lock', r'$^'), code))
        features['count_promise'] = len(re.findall(lang_patterns.get('promise', r'$^'), code))

        # Determine if uses concurrency
        concurrency_indicators = [
            features['count_async'],
            features['count_await'],
            features['count_thread'],
            features['count_mutex'],
            features['count_promise']
        ]

        features['uses_concurrency'] = 1 if any(concurrency_indicators) else 0

        # Determine concurrency pattern
        if features['count_async'] > 0 or features['count_await'] > 0:
            features['concurrency_pattern'] = 'async_await'
        elif features['count_thread'] > 0:
            features['concurrency_pattern'] = 'threading'
        elif features['count_promise'] > 0:
            features['concurrency_pattern'] = 'event_loop'
        else:
            features['concurrency_pattern'] = 'none'

        return features


if __name__ == "__main__":
    # Example usage
    analyzer = ASTAnalyzer()
    concurrency_extractor = ConcurrencyFeatureExtractor()

    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

    features = analyzer.analyze(sample_code, 'Python')
    print("AST Features:", features)

    conc_features = concurrency_extractor.extract(sample_code, 'Python')
    print("Concurrency Features:", conc_features)
