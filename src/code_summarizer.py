"""
Code Summarization Module
Generates natural language summaries of code snippets
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSummarizer:
    """Summarizes code into natural language descriptions"""

    def __init__(self, model_name="Salesforce/codet5-base"):
        """
        Initialize the code summarizer
        
        Args:
            model_name: HuggingFace model for code summarization
        """
        logger.info(f"Loading code summarization model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}, using rule-based summarization")
            self.model = None
            self.tokenizer = None

    def summarize_code(self, code: str, language: str = "python", 
                      max_length: int = 128) -> str:
        """
        Generate a natural language summary of code
        
        Args:
            code: Source code to summarize
            language: Programming language of the code
            max_length: Maximum length of summary
            
        Returns:
            Natural language summary
        """
        logger.info(f"Summarizing {language} code")
        
        if self.model is None:
            # Fallback to rule-based summarization
            return self._rule_based_summarization(code, language)
        
        try:
            # Prepare input
            input_text = f"summarize {language}: {code}"
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            # Generate summary
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing code: {e}")
            return self._rule_based_summarization(code, language)

    def _rule_based_summarization(self, code: str, language: str) -> str:
        """Rule-based code summarization as fallback"""
        
        summary_parts = []
        
        # Extract function/class names
        if language.lower() == "python":
            # Look for function definitions
            functions = re.findall(r'def\s+(\w+)\s*\(', code)
            if functions:
                summary_parts.append(f"Defines function(s): {', '.join(functions)}")
            
            # Look for class definitions
            classes = re.findall(r'class\s+(\w+)', code)
            if classes:
                summary_parts.append(f"Defines class(es): {', '.join(classes)}")
            
            # Look for imports
            imports = re.findall(r'import\s+(\w+)|from\s+(\w+)', code)
            if imports:
                imp_list = [imp[0] or imp[1] for imp in imports]
                summary_parts.append(f"Uses modules: {', '.join(set(imp_list))}")
        
        elif language.lower() == "java":
            # Look for class definitions
            classes = re.findall(r'class\s+(\w+)', code)
            if classes:
                summary_parts.append(f"Defines class: {classes[0]}")
            
            # Look for methods
            methods = re.findall(r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(', code)
            if methods:
                summary_parts.append(f"Contains methods: {', '.join(methods[:3])}")
        
        elif language.lower() == "javascript":
            # Look for function definitions
            functions = re.findall(r'function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\(|async)', code)
            if functions:
                func_list = [f[0] or f[1] for f in functions]
                summary_parts.append(f"Defines function(s): {', '.join(func_list)}")
        
        # Extract comments
        comments = self._extract_comments(code, language)
        if comments:
            summary_parts.append(f"Documentation: {comments[0][:100]}")
        
        # Check for common patterns
        if "if" in code:
            summary_parts.append("Contains conditional logic")
        if "for" in code or "while" in code:
            summary_parts.append("Contains loops")
        if "try" in code or "except" in code or "catch" in code:
            summary_parts.append("Includes error handling")
        
        if not summary_parts:
            return "Code snippet with implementation details"
        
        return ". ".join(summary_parts) + "."

    def _extract_comments(self, code: str, language: str) -> list:
        """Extract comments from code"""
        comments = []
        
        if language.lower() == "python":
            # Single line comments
            comments.extend(re.findall(r'#\s*(.+)', code))
            # Docstrings
            comments.extend(re.findall(r'"""(.+?)"""', code, re.DOTALL))
            comments.extend(re.findall(r"'''(.+?)'''", code, re.DOTALL))
        
        elif language.lower() in ["java", "javascript"]:
            # Single line comments
            comments.extend(re.findall(r'//\s*(.+)', code))
            # Multi-line comments
            comments.extend(re.findall(r'/\*(.+?)\*/', code, re.DOTALL))
        
        return [c.strip() for c in comments if c.strip()]

    def summarize_with_details(self, code: str, language: str = "python") -> dict:
        """
        Generate detailed summary with multiple aspects
        
        Args:
            code: Source code to summarize
            language: Programming language
            
        Returns:
            Dictionary with different summary aspects
        """
        summary = {
            "overview": self.summarize_code(code, language),
            "complexity": self._assess_complexity(code),
            "key_operations": self._extract_key_operations(code, language),
            "dependencies": self._extract_dependencies(code, language)
        }
        
        return summary

    def _assess_complexity(self, code: str) -> str:
        """Assess code complexity"""
        lines = len(code.split('\n'))
        
        if lines < 10:
            return "Simple (< 10 lines)"
        elif lines < 50:
            return "Moderate (10-50 lines)"
        elif lines < 100:
            return "Complex (50-100 lines)"
        else:
            return "Very Complex (> 100 lines)"

    def _extract_key_operations(self, code: str, language: str) -> list:
        """Extract key operations from code"""
        operations = []
        
        # Common operations across languages
        if "read" in code.lower() or "open" in code.lower():
            operations.append("File I/O")
        if "http" in code.lower() or "request" in code.lower():
            operations.append("Network requests")
        if "database" in code.lower() or "sql" in code.lower():
            operations.append("Database operations")
        if "sort" in code.lower():
            operations.append("Sorting")
        if "search" in code.lower() or "find" in code.lower():
            operations.append("Searching")
        
        return operations if operations else ["General computation"]

    def _extract_dependencies(self, code: str, language: str) -> list:
        """Extract dependencies/imports"""
        dependencies = []
        
        if language.lower() == "python":
            imports = re.findall(r'import\s+([\w.]+)|from\s+([\w.]+)', code)
            dependencies = list(set([imp[0] or imp[1] for imp in imports]))
        
        elif language.lower() == "java":
            imports = re.findall(r'import\s+([\w.]+);', code)
            dependencies = list(set(imports))
        
        elif language.lower() == "javascript":
            imports = re.findall(r'import\s+.*?from\s+["\'](.+?)["\']|require\(["\'](.+?)["\']\)', code)
            dependencies = list(set([imp[0] or imp[1] for imp in imports]))
        
        return dependencies

    def batch_summarize(self, code_snippets: list, language: str = "python") -> list:
        """
        Summarize multiple code snippets
        
        Args:
            code_snippets: List of code strings
            language: Programming language
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i, code in enumerate(code_snippets, 1):
            logger.info(f"Summarizing snippet {i}/{len(code_snippets)}")
            summary = self.summarize_code(code, language)
            summaries.append(summary)
        
        return summaries


# Simple usage example
if __name__ == "__main__":
    summarizer = CodeSummarizer()
    
    # Example 1: Python function
    python_code = """
def fibonacci(n):
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    summary = summarizer.summarize_code(python_code, "python")
    print("Summary:", summary)
    print("\n" + "="*80 + "\n")
    
    # Example 2: Detailed summary
    detailed = summarizer.summarize_with_details(python_code, "python")
    print("Detailed Summary:")
    for key, value in detailed.items():
        print(f"  {key}: {value}")
