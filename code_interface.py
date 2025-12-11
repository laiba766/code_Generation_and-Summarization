"""
Interactive Code Generation and Summarization Interface
Allows users to generate code from English descriptions and summarize existing code
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.append('src')

from code_generator import CodeGenerator
from code_summarizer import CodeSummarizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_code_interactive(generator, language="python"):
    """Interactive code generation"""
    print("\n" + "="*80)
    print("CODE GENERATION MODE")
    print("="*80)
    print("Enter your description in English (or 'quit' to exit)")
    print("Example: 'Function to calculate the sum of a list of numbers'")
    print("-"*80)
    
    while True:
        description = input("\nDescribe what you want to create: ").strip()
        
        if description.lower() in ['quit', 'exit', 'q']:
            break
        
        if not description:
            print("Please enter a description!")
            continue
        
        print(f"\n🤖 Generating {language} code...")
        code = generator.generate_code(description, language)
        
        print("\n✨ Generated Code:")
        print("-"*80)
        print(code)
        print("-"*80)
        
        # Ask if user wants variations
        choice = input("\nGenerate variations? (y/n): ").strip().lower()
        if choice == 'y':
            num = input("How many variations? (1-5): ").strip()
            try:
                num = int(num)
                if 1 <= num <= 5:
                    variations = generator.generate_multiple_variations(
                        description, language, num
                    )
                    for i, var in enumerate(variations, 1):
                        print(f"\n✨ Variation {i}:")
                        print("-"*80)
                        print(var)
                        print("-"*80)
            except ValueError:
                print("Invalid number!")


def summarize_code_interactive(summarizer, language="python"):
    """Interactive code summarization"""
    print("\n" + "="*80)
    print("CODE SUMMARIZATION MODE")
    print("="*80)
    print("Enter your code (press Ctrl+D or Ctrl+Z when done, or 'quit' to exit)")
    print("-"*80)
    
    while True:
        print("\nPaste your code (multi-line supported):")
        print("(Type 'END' on a new line when finished, or 'quit' to exit)")
        
        lines = []
        while True:
            try:
                line = input()
                if line.strip().lower() == 'quit':
                    return
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break
        
        code = '\n'.join(lines)
        
        if not code.strip():
            print("No code entered!")
            continue
        
        print("\n🤖 Analyzing code...")
        
        # Basic summary
        summary = summarizer.summarize_code(code, language)
        print("\n📝 Summary:")
        print("-"*80)
        print(summary)
        print("-"*80)
        
        # Ask for detailed summary
        choice = input("\nShow detailed analysis? (y/n): ").strip().lower()
        if choice == 'y':
            detailed = summarizer.summarize_with_details(code, language)
            print("\n📊 Detailed Analysis:")
            print("-"*80)
            for key, value in detailed.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
            print("-"*80)


def process_file(file_path, mode, generator, summarizer, language):
    """Process a file for generation or summarization"""
    
    if mode == "summarize":
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        print(f"\n📄 Reading file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        print("🤖 Generating summary...")
        detailed = summarizer.summarize_with_details(code, language)
        
        print("\n" + "="*80)
        print(f"SUMMARY OF: {file_path}")
        print("="*80)
        for key, value in detailed.items():
            print(f"\n{key.replace('_', ' ').title()}:")
            print(f"  {value}")
        print("\n" + "="*80)
        
        # Save summary
        summary_file = file_path.replace('.py', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Summary of {file_path}\n")
            f.write("="*80 + "\n\n")
            for key, value in detailed.items():
                f.write(f"{key.replace('_', ' ').title()}:\n")
                f.write(f"  {value}\n\n")
        
        print(f"\n✅ Summary saved to: {summary_file}")
    
    elif mode == "generate":
        print(f"\n📄 Reading description from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            description = f.read().strip()
        
        print("🤖 Generating code...")
        code = generator.generate_code(description, language)
        
        print("\n✨ Generated Code:")
        print("="*80)
        print(code)
        print("="*80)
        
        # Save generated code
        output_ext = {
            'python': '.py',
            'java': '.java',
            'javascript': '.js'
        }.get(language.lower(), '.txt')
        
        output_file = file_path.replace('.txt', '_generated' + output_ext)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"\n✅ Code saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate code from English or summarize existing code"
    )
    parser.add_argument(
        '--mode',
        choices=['generate', 'summarize', 'both'],
        default='both',
        help='Operation mode: generate code, summarize code, or both'
    )
    parser.add_argument(
        '--language',
        choices=['python', 'java', 'javascript'],
        default='python',
        help='Programming language to use'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='File to process (description for generation, code for summarization)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 CODE GENERATION & SUMMARIZATION TOOL")
    print("="*80)
    print(f"Language: {args.language.upper()}")
    print(f"Mode: {args.mode.upper()}")
    print("="*80)
    
    # Initialize models
    generator = None
    summarizer = None
    
    if args.mode in ['generate', 'both']:
        print("\n⏳ Loading code generation model...")
        generator = CodeGenerator()
    
    if args.mode in ['summarize', 'both']:
        print("\n⏳ Loading code summarization model...")
        summarizer = CodeSummarizer()
    
    # File mode
    if args.file:
        if args.mode == 'generate':
            process_file(args.file, 'generate', generator, None, args.language)
        elif args.mode == 'summarize':
            process_file(args.file, 'summarize', None, summarizer, args.language)
        else:
            print("Error: Please specify --mode when using --file")
        return
    
    # Interactive mode
    if args.interactive or not args.file:
        if args.mode == 'both':
            while True:
                print("\n" + "="*80)
                print("Choose an option:")
                print("  1. Generate code from description")
                print("  2. Summarize existing code")
                print("  3. Exit")
                print("-"*80)
                
                choice = input("Enter your choice (1-3): ").strip()
                
                if choice == '1':
                    generate_code_interactive(generator, args.language)
                elif choice == '2':
                    summarize_code_interactive(summarizer, args.language)
                elif choice == '3':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("Invalid choice!")
        
        elif args.mode == 'generate':
            generate_code_interactive(generator, args.language)
        
        elif args.mode == 'summarize':
            summarize_code_interactive(summarizer, args.language)


if __name__ == "__main__":
    main()
