"""
Data Collection Module
Collects code snippets from CodeSearchNet and GitHub repositories
"""

import os
import json
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects code snippets from various sources"""

    def __init__(self, config: Dict):
        self.config = config
        self.languages = config['data']['languages']
        self.min_lines = config['data']['sampling']['min_lines']
        self.max_lines = config['data']['sampling']['max_lines']
        self.target_per_lang = config['data']['sampling']['target_functions_per_lang']
        self.exclude_patterns = config['data']['sampling']['exclude_patterns']

    def is_excluded(self, file_path: str) -> bool:
        """Check if file should be excluded based on patterns"""
        for pattern in self.exclude_patterns:
            pattern_regex = pattern.replace('**/', '').replace('*', '.*')
            if re.search(pattern_regex, file_path):
                return True
        return False

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'Python',
            '.java': 'Java',
            '.js': 'JavaScript',
            '.rs': 'Rust'
        }
        ext = Path(file_path).suffix
        return ext_map.get(ext)

    def is_test_file(self, file_path: str, content: str) -> bool:
        """Heuristic to detect test files"""
        file_name = os.path.basename(file_path).lower()

        # Check filename
        test_indicators = ['test', 'spec', '_test', 'test_']
        if any(indicator in file_name for indicator in test_indicators):
            return True

        # Check content for common test frameworks
        test_patterns = [
            r'import\s+unittest',
            r'from\s+unittest',
            r'import\s+pytest',
            r'@Test',
            r'describe\(',
            r'it\(',
            r'#\[test\]',
            r'#\[cfg\(test\)\]'
        ]

        for pattern in test_patterns:
            if re.search(pattern, content):
                return True

        return False

    def extract_functions_python(self, content: str, file_path: str) -> List[Dict]:
        """Extract Python functions from source code"""
        import ast

        functions = []
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line

                    # Extract function source
                    lines = content.split('\n')
                    func_code = '\n'.join(lines[start_line-1:end_line])

                    # Count non-empty, non-comment lines
                    code_lines = [l.strip() for l in func_code.split('\n')
                                  if l.strip() and not l.strip().startswith('#')]

                    if self.min_lines <= len(code_lines) <= self.max_lines:
                        functions.append({
                            'function_name': node.name,
                            'code': func_code,
                            'start_line': start_line,
                            'end_line': end_line,
                            'file_path': file_path,
                            'language': 'Python'
                        })

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")

        return functions

    def extract_functions_java(self, content: str, file_path: str) -> List[Dict]:
        """Extract Java methods using regex patterns"""
        functions = []

        # Pattern for Java methods (simplified)
        method_pattern = r'(public|private|protected)\s+(?:static\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'

        lines = content.split('\n')

        for match in re.finditer(method_pattern, content):
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1

            # Find matching closing brace
            brace_count = 1
            pos = match.end()
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1

            end_line = content[:pos].count('\n') + 1
            func_code = '\n'.join(lines[start_line-1:end_line])

            # Count non-empty, non-comment lines
            code_lines = [l.strip() for l in func_code.split('\n')
                          if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('/*')]

            if self.min_lines <= len(code_lines) <= self.max_lines:
                functions.append({
                    'function_name': match.group(2),
                    'code': func_code,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': file_path,
                    'language': 'Java'
                })

        return functions

    def extract_functions_javascript(self, content: str, file_path: str) -> List[Dict]:
        """Extract JavaScript functions using regex patterns"""
        functions = []

        # Patterns for different function declarations
        patterns = [
            r'function\s+(\w+)\s*\([^)]*\)\s*\{',  # function name() {}
            r'(?:const|let|var)\s+(\w+)\s*=\s*function\s*\([^)]*\)\s*\{',  # const name = function() {}
            r'(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{',  # const name = () => {}
            r'(\w+)\s*:\s*function\s*\([^)]*\)\s*\{',  # name: function() {}
        ]

        lines = content.split('\n')

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1

                # Find matching closing brace
                brace_count = 1
                pos = match.end()
                while pos < len(content) and brace_count > 0:
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                    pos += 1

                end_line = content[:pos].count('\n') + 1
                func_code = '\n'.join(lines[start_line-1:end_line])

                # Count non-empty, non-comment lines
                code_lines = [l.strip() for l in func_code.split('\n')
                              if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('/*')]

                if self.min_lines <= len(code_lines) <= self.max_lines:
                    functions.append({
                        'function_name': match.group(1),
                        'code': func_code,
                        'start_line': start_line,
                        'end_line': end_line,
                        'file_path': file_path,
                        'language': 'JavaScript'
                    })

        return functions

    def extract_functions_rust(self, content: str, file_path: str) -> List[Dict]:
        """Extract Rust functions using regex patterns"""
        functions = []

        # Pattern for Rust functions
        fn_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{'

        lines = content.split('\n')

        for match in re.finditer(fn_pattern, content):
            start_pos = match.start()
            start_line = content[:start_pos].count('\n') + 1

            # Find matching closing brace
            brace_count = 1
            pos = match.end()
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1

            end_line = content[:pos].count('\n') + 1
            func_code = '\n'.join(lines[start_line-1:end_line])

            # Count non-empty, non-comment lines
            code_lines = [l.strip() for l in func_code.split('\n')
                          if l.strip() and not l.strip().startswith('//') and not l.strip().startswith('/*')]

            if self.min_lines <= len(code_lines) <= self.max_lines:
                functions.append({
                    'function_name': match.group(1),
                    'code': func_code,
                    'start_line': start_line,
                    'end_line': end_line,
                    'file_path': file_path,
                    'language': 'Rust'
                })

        return functions

    def extract_functions(self, content: str, file_path: str, language: str) -> List[Dict]:
        """Extract functions based on language"""
        if language == 'Python':
            return self.extract_functions_python(content, file_path)
        elif language == 'Java':
            return self.extract_functions_java(content, file_path)
        elif language == 'JavaScript':
            return self.extract_functions_javascript(content, file_path)
        elif language == 'Rust':
            return self.extract_functions_rust(content, file_path)
        else:
            return []

    def process_file(self, file_path: str, content: str) -> List[Dict]:
        """Process a single file and extract functions"""
        if self.is_excluded(file_path):
            return []

        language = self.detect_language(file_path)
        if not language or language not in self.languages:
            return []

        if self.is_test_file(file_path, content):
            return []

        return self.extract_functions(content, file_path, language)

    def collect_from_directory(self, directory: str) -> Dict[str, List[Dict]]:
        """Collect functions from a directory"""
        logger.info(f"Collecting functions from {directory}")

        collected = {lang: [] for lang in self.languages}

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                language = self.detect_language(file_path)

                if language and language in self.languages:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            functions = self.process_file(file_path, content)
                            collected[language].extend(functions)

                            # Stop if we have enough
                            if len(collected[language]) >= self.target_per_lang:
                                break
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")

        return collected

    def save_dataset(self, functions: Dict[str, List[Dict]], output_dir: str):
        """Save collected functions to disk"""
        os.makedirs(output_dir, exist_ok=True)

        for language, funcs in functions.items():
            output_file = os.path.join(output_dir, f'{language.lower()}_functions.jsonl')

            with open(output_file, 'w', encoding='utf-8') as f:
                for func in funcs:
                    f.write(json.dumps(func) + '\n')

            logger.info(f"Saved {len(funcs)} {language} functions to {output_file}")


if __name__ == "__main__":
    import yaml

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    collector = DataCollector(config)

    # Example usage - collect from a directory
    # functions = collector.collect_from_directory('/path/to/code/repository')
    # collector.save_dataset(functions, 'data/raw')
