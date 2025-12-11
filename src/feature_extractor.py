"""
Feature Extractor Module
Combines AST analysis, complexity metrics, and PL features
"""

import json
import pandas as pd
from typing import Dict, List, Any
import logging
from ast_parser import ASTAnalyzer, ConcurrencyFeatureExtractor
from complexity_metrics import ComplexityCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLFeatureExtractor:
    """Extracts Programming Language level features"""

    def __init__(self):
        self.language_features = {
            'Python': {
                'is_static_typed': 0,
                'memory_model': 'managed',
                'paradigm_imperative': 1,
                'paradigm_oop': 1,
                'paradigm_functional': 1,
                'concurrency_model': 'async_await'
            },
            'Java': {
                'is_static_typed': 1,
                'memory_model': 'managed',
                'paradigm_imperative': 1,
                'paradigm_oop': 1,
                'paradigm_functional': 0,
                'concurrency_model': 'threading'
            },
            'JavaScript': {
                'is_static_typed': 0,
                'memory_model': 'managed',
                'paradigm_imperative': 1,
                'paradigm_oop': 1,
                'paradigm_functional': 1,
                'concurrency_model': 'event_loop'
            },
            'Rust': {
                'is_static_typed': 1,
                'memory_model': 'ownership',
                'paradigm_imperative': 1,
                'paradigm_oop': 0,
                'paradigm_functional': 1,
                'concurrency_model': 'threading'
            }
        }

    def extract(self, language: str) -> Dict[str, Any]:
        """Extract PL-level features for a language"""
        if language not in self.language_features:
            logger.warning(f"Unknown language: {language}")
            return {
                'is_static_typed': 0,
                'memory_model': 'unknown',
                'paradigm_imperative': 0,
                'paradigm_oop': 0,
                'paradigm_functional': 0,
                'concurrency_model': 'none'
            }

        return self.language_features[language].copy()

    def encode_categorical(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """One-hot encode categorical features"""
        encoded = features.copy()

        # Memory model encoding
        memory_models = ['manual', 'managed', 'ownership', 'unknown']
        for model in memory_models:
            encoded[f'memory_model_{model}'] = 1 if features['memory_model'] == model else 0
        del encoded['memory_model']

        # Concurrency model encoding
        concurrency_models = ['none', 'threading', 'async_await', 'event_loop']
        for model in concurrency_models:
            encoded[f'concurrency_model_{model}'] = 1 if features['concurrency_model'] == model else 0
        del encoded['concurrency_model']

        return encoded


class FeatureExtractor:
    """Main feature extraction pipeline"""

    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
        self.complexity_calculator = ComplexityCalculator()
        self.concurrency_extractor = ConcurrencyFeatureExtractor()
        self.pl_extractor = PLFeatureExtractor()

    def extract_function_features(self, code: str, language: str, metadata: Dict = None) -> Dict[str, Any]:
        """Extract all features for a single function"""
        features = {}

        # Add metadata
        if metadata:
            features.update(metadata)
        features['language'] = language

        # PL-level features
        pl_features = self.pl_extractor.extract(language)
        pl_encoded = self.pl_extractor.encode_categorical(pl_features)
        features.update(pl_encoded)

        # AST features
        try:
            ast_features = self.ast_analyzer.analyze(code, language)
            if ast_features:
                features.update(ast_features)
        except Exception as e:
            logger.warning(f"Error extracting AST features: {e}")

        # Complexity metrics
        try:
            complexity_features = self.complexity_calculator.calculate_all(code, language)
            features.update(complexity_features)
        except Exception as e:
            logger.warning(f"Error calculating complexity: {e}")

        # Concurrency features
        try:
            concurrency_features = self.concurrency_extractor.extract(code, language)
            features.update(concurrency_features)
        except Exception as e:
            logger.warning(f"Error extracting concurrency features: {e}")

        # Size and formatting controls
        lines = code.split('\n')
        features['num_parameters'] = code.count(',') + 1 if '(' in code else 0
        features['num_local_variables'] = len(set([
            word for line in lines
            for word in line.split()
            if word.isidentifier()
        ]))

        return features

    def process_dataset(self, input_file: str, output_file: str):
        """Process a dataset of functions and extract features"""
        logger.info(f"Processing dataset: {input_file}")

        all_features = []

        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    code = data.get('code', '')
                    language = data.get('language', '')

                    metadata = {
                        'function_id': f"{language}_{line_num}",
                        'function_name': data.get('function_name', ''),
                        'file_path': data.get('file_path', ''),
                        'repo_name': data.get('repo_name', ''),
                        'start_line': data.get('start_line', 0),
                        'end_line': data.get('end_line', 0)
                    }

                    features = self.extract_function_features(code, language, metadata)
                    all_features.append(features)

                    if line_num % 100 == 0:
                        logger.info(f"Processed {line_num} functions")

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} function features to {output_file}")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        sample_features = self.extract_function_features("def f(): pass", "Python")
        return list(sample_features.keys())


if __name__ == "__main__":
    extractor = FeatureExtractor()

    # Example usage
    sample_code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

    features = extractor.extract_function_features(sample_code, 'Python',
                                                    {'function_name': 'quicksort'})

    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
