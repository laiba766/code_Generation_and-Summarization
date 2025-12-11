"""
Code Generation Module
Generates code from natural language descriptions using transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates code from natural language descriptions"""

    def __init__(self, model_name="Salesforce/codegen-350M-mono"):
        """
        Initialize the code generator
        
        Args:
            model_name: HuggingFace model for code generation
        """
        logger.info(f"Loading code generation model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load {model_name}, using simple template-based generation")
            self.model = None
            self.tokenizer = None

    def generate_code(self, description: str, language: str = "python", 
                     max_length: int = 200, temperature: float = 0.7) -> str:
        """
        Generate code from natural language description
        
        Args:
            description: Natural language description of what the code should do
            language: Target programming language
            max_length: Maximum length of generated code
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated code as a string
        """
        logger.info(f"Generating {language} code for: {description}")
        
        if self.model is None:
            # Fallback to template-based generation
            return self._template_based_generation(description, language)
        
        try:
            # Format prompt for code generation
            prompt = self._format_prompt(description, language)
            
            # Generate code
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            if language.lower() == "python":
                # Remove the prompt and keep only the function body
                if "def " in generated_code:
                    # Find the first def after the comment
                    parts = generated_code.split('\n')
                    code_lines = []
                    started = False
                    for line in parts:
                        if line.strip().startswith('def ') or started:
                            started = True
                            code_lines.append(line)
                    generated_code = '\n'.join(code_lines)
            
            return generated_code.strip()
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return self._template_based_generation(description, language)

    def _format_prompt(self, description: str, language: str) -> str:
        """Format the prompt for the model"""
        if language.lower() == "python":
            return f"# {description}\ndef "
        elif language.lower() == "java":
            return f"// {description}\npublic "
        elif language.lower() == "javascript":
            return f"// {description}\nfunction "
        else:
            return f"// {description}\n"

    def _template_based_generation(self, description: str, language: str) -> str:
        """Simple template-based code generation as fallback"""
        
        templates = {
            "python": f'''def generated_function():
    """
    {description}
    """
    # TODO: Implement the logic for: {description}
    pass
''',
            "java": f'''public class GeneratedClass {{
    // {description}
    public void generatedMethod() {{
        // TODO: Implement the logic for: {description}
    }}
}}
''',
            "javascript": f'''// {description}
function generatedFunction() {{
    // TODO: Implement the logic for: {description}
}}
'''
        }
        
        return templates.get(language.lower(), f"// {description}\n// TODO: Implement this")

    def generate_multiple_variations(self, description: str, language: str = "python", 
                                    num_variations: int = 3) -> list:
        """
        Generate multiple code variations for the same description
        
        Args:
            description: Natural language description
            language: Target programming language
            num_variations: Number of variations to generate
            
        Returns:
            List of generated code strings
        """
        variations = []
        
        for i in range(num_variations):
            logger.info(f"Generating variation {i+1}/{num_variations}")
            # Use different temperatures for diversity
            temp = 0.6 + (i * 0.2)
            code = self.generate_code(description, language, temperature=temp)
            variations.append(code)
        
        return variations

    def generate_from_examples(self, description: str, examples: list, 
                              language: str = "python") -> str:
        """
        Generate code based on description and example inputs/outputs
        
        Args:
            description: What the function should do
            examples: List of (input, output) tuples
            language: Target programming language
            
        Returns:
            Generated code
        """
        # Create a detailed prompt with examples
        examples_str = "\n".join([
            f"Input: {inp} -> Output: {out}" 
            for inp, out in examples
        ])
        
        detailed_description = f"{description}\n\nExamples:\n{examples_str}"
        
        return self.generate_code(detailed_description, language)


# Simple usage example
if __name__ == "__main__":
    generator = CodeGenerator()
    
    # Example 1: Basic function
    description = "Function to calculate the factorial of a number"
    code = generator.generate_code(description, "python")
    print("Generated Code:")
    print(code)
    print("\n" + "="*80 + "\n")
    
    # Example 2: Multiple variations
    description = "Function to check if a string is a palindrome"
    variations = generator.generate_multiple_variations(description, "python", num_variations=2)
    for i, var in enumerate(variations, 1):
        print(f"Variation {i}:")
        print(var)
        print("\n" + "="*80 + "\n")
