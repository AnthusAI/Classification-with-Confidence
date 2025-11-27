#!/usr/bin/env python3
"""
Educational Logprob Demonstration CLI Tool

This tool shows the detailed token probability analysis that language models
perform internally, displaying both raw log-probabilities and converted
probabilities in the format used in the README examples.

Usage:
    python logprob_demo_cli.py "Your text here"
    python logprob_demo_cli.py "Your text here" --model finetuned
    python logprob_demo_cli.py "Your text here" --model base --top-k 10

Examples:
    python logprob_demo_cli.py "I love this movie!"
    python logprob_demo_cli.py "Best worst thing ever" --model finetuned
"""

import sys
import argparse
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Tuple
from logprobs_confidence import TransformerLogprobsClassifier
from classification_config import ClassificationConfig, ClassificationMode


def format_logprob_table(result: Dict[str, Any], top_k: int = 8) -> str:
    """
    Format the logprob results into the README-style table format.
    Shows log-probabilities first (what the model actually computes),
    then probabilities (derived from log-probs).
    """
    if 'error' in result:
        return f"❌ Error: {result['error']}"
    
    # Get token details and sort by probability (descending)
    token_details = result.get('token_details', {})
    if not token_details:
        return "❌ No token details available"
    
    # Sort tokens by probability (highest first)
    sorted_tokens = sorted(
        token_details.items(), 
        key=lambda x: x[1]['prob'], 
        reverse=True
    )
    
    # Take top-k tokens
    top_tokens = sorted_tokens[:top_k]
    
    # Build the table
    lines = []
    lines.append("FIRST TOKEN PROBABILITY DISTRIBUTION:")
    lines.append("Rank  Token        Log-Prob    Probability    Percentage")
    lines.append("-" * 55)
    
    for rank, (token, details) in enumerate(top_tokens, 1):
        # Clean up token display (remove extra spaces, show quotes)
        display_token = f'"{token.strip():<8}"' if token.strip() else f'"{token:<8}"'
        log_prob = details['logprob']
        probability = details['prob']
        percentage = probability * 100
        
        lines.append(
            f"{rank:>2}.   {display_token:<12} {log_prob:>8.3f}    {probability:>10.6f}    {percentage:>6.2f}%"
        )
    
    return "\n".join(lines)


def format_aggregation_summary(result: Dict[str, Any]) -> str:
    """Format the sentiment aggregation summary."""
    if 'error' in result:
        return ""
    
    sentiment_probs = result.get('sentiment_probabilities', {})
    prediction = result.get('prediction', 'unknown')
    confidence = result.get('confidence', 0.0)
    
    lines = []
    lines.append("\nSENTIMENT CLASSIFICATION AGGREGATION:")
    
    # Show all sentiment variants
    pos_prob = sentiment_probs.get('positive', 0.0) * 100
    neg_prob = sentiment_probs.get('negative', 0.0) * 100
    
    lines.append(f'All "YES" variants (Yes, yes, YES, y, Y):     {pos_prob:>5.2f}%')
    lines.append(f'All "NO" variants (No, no, NO, n, N):        {neg_prob:>5.2f}%')
    lines.append("")
    lines.append(f"FINAL PREDICTION: {prediction.upper()}")
    lines.append(f"CONFIDENCE: {confidence*100:.2f}%")
    
    return "\n".join(lines)


def generate_multi_token_markdown(result: Dict[str, Any], model_type: str) -> str:
    """Generate markdown report for multi-token analysis."""
    if 'error' in result:
        return f"# Error\n\n{result['error']}"

    model_name = "Fine-tuned Model" if model_type == "finetuned" else "Base Model"
    token_analyses = result.get('token_analyses', [])

    md_lines = []
    md_lines.append(f"# Multi-Token Log Probability Analysis - {model_name}")
    md_lines.append(f"\n**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"\n**Model:** {result.get('model', 'Unknown')}")
    md_lines.append(f"\n**Prompt:** `{result.get('prompt', '')}`")
    md_lines.append(f"\n**Completion:** `{result.get('generated_text', '')}`")
    md_lines.append(f"\n**Total Tokens:** {len(token_analyses)}")
    md_lines.append("\n---\n")

    if not token_analyses:
        md_lines.append("No token analyses available.")
        return "\n".join(md_lines)

    # Create single table with one column per token
    # Header row with token positions
    header = "| Rank |"
    for i, analysis in enumerate(token_analyses, 1):
        header += f" Token {i} |"
    md_lines.append(header)

    # Separator row
    separator = "|------|"
    for _ in token_analyses:
        separator += "--------|"
    md_lines.append(separator)

    # Get max number of top tokens to show (12 as requested)
    max_tokens = 12

    # Create rows for each rank
    for rank in range(1, max_tokens + 1):
        row = f"| {rank} |"

        for analysis in token_analyses:
            top_tokens = analysis.get('top_tokens', [])
            if rank <= len(top_tokens):
                token_info = top_tokens[rank - 1]
                token = token_info['token'].replace('|', '\\|').strip()
                percentage = token_info['percentage']

                # Combine token and percentage in one cell, centered
                cell_content = f"**{token}**<br/><small>{percentage:.1f}%</small>"
                row += f" {cell_content} |"
            else:
                row += " |"

        md_lines.append(row)

    return "\n".join(md_lines)


def format_multi_token_console_output(result: Dict[str, Any]) -> str:
    """Format multi-token results for console display."""
    if 'error' in result:
        return f"❌ Error: {result['error']}"

    lines = []
    lines.append("📝 MULTI-TOKEN GENERATION ANALYSIS:")
    lines.append(f"Generated Text: \"{result.get('generated_text', '')}\"")
    lines.append(f"Total Tokens: {len(result.get('token_analyses', []))}")
    lines.append("")

    # Show analysis for each token
    for analysis in result.get('token_analyses', []):
        step = analysis['step']
        generated_token = analysis['generated_token']

        lines.append(f"TOKEN {step}: \"{generated_token}\"")
        lines.append("Rank  Token        Log-Prob    Probability    Percentage")
        lines.append("-" * 55)

        for token_info in analysis['top_tokens']:
            rank = token_info['rank']
            token = f'"{token_info["token"]:<8}"'
            log_prob = token_info['log_probability']
            probability = token_info['probability']
            percentage = token_info['percentage']
            selected = " ← SELECTED" if token_info['is_generated'] else ""

            lines.append(
                f"{rank:>2}.   {token:<12} {log_prob:>8.3f}    {probability:>10.6f}    {percentage:>6.2f}%{selected}"
            )

        lines.append("")

    return "\n".join(lines)


def save_markdown_report(content: str, filename: str) -> str:
    """Save markdown content to temp folder and return the path."""
    # Create temp directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Create full path
    filepath = os.path.join(temp_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return os.path.abspath(filepath)


def print_model_header(model_type: str, text: str, raw_prompt: bool = False):
    """Print a formatted header for the model analysis."""
    model_name = "FINE-TUNED MODEL" if model_type == "finetuned" else "BASE MODEL"
    print(f"\n{'='*60}")
    print(f"🧠 {model_name} ANALYSIS")
    print(f"{'='*60}")
    if raw_prompt:
        print(f'RAW PROMPT: "{text}"')
    else:
        print(f'INPUT: "Is this text positive in sentiment? Answer yes or no."')
        print(f'TEXT TO CLASSIFY: "{text}"')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Educational tool showing detailed logprob analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python logprob_demo_cli.py "I love this movie!"
  python logprob_demo_cli.py "Best worst thing ever" --model finetuned
  python logprob_demo_cli.py "This is okay" --top-k 10
  python logprob_demo_cli.py "Think step by step. Is this positive? Answer:" --multiple-tokens 20
        """
    )
    
    parser.add_argument('text', help='Text to analyze')
    parser.add_argument(
        '--model', 
        choices=['base', 'finetuned'], 
        default='base',
        help='Model to use: base or finetuned (default: base)'
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=8,
        help='Number of top tokens to show (default: 8)'
    )
    parser.add_argument(
        '--quiet', '-q', 
        action='store_true',
        help='Minimal output (just prediction and confidence)'
    )
    parser.add_argument(
        '--raw-prompt', '-r',
        action='store_true',
        help='Use text as raw prompt instead of wrapping in classification question'
    )
    parser.add_argument(
        '--multiple-tokens', '-m',
        action='store_true',
        help='Show logprobs for all tokens in the natural completion (creates markdown report)'
    )
    parser.add_argument(
        '--chain-of-thought', '-c',
        action='store_true',
        help='Use chain-of-thought prompt format (explanation followed by YES/NO)'
    )
    parser.add_argument(
        '--classification-mode',
        choices=['first-token', 'last-token'],
        default='first-token',
        help='Classification mode: first-token (direct) or last-token (chain-of-thought)'
    )
    
    args = parser.parse_args()
    
    if not args.text.strip():
        print("❌ Error: Empty text provided")
        sys.exit(1)
    
    try:
        # Create classification configuration
        if args.chain_of_thought or args.classification_mode == 'last-token':
            mode = ClassificationMode.LAST_TOKEN
        else:
            mode = ClassificationMode.FIRST_TOKEN

        config = ClassificationConfig(mode=mode)

        # Initialize the appropriate classifier
        if args.model == 'finetuned':
            fine_tuned_path = "./fine_tuned_sentiment_model"
            import os
            if not os.path.exists(fine_tuned_path):
                print("❌ Error: Fine-tuned model not found!")
                print("💡 Run 'python fine_tune_model.py' first to create the fine-tuned model.")
                sys.exit(1)
            classifier = TransformerLogprobsClassifier(fine_tuned_path=fine_tuned_path, config=config)
        else:
            classifier = TransformerLogprobsClassifier(config=config)
        
        # Test model availability
        if not classifier.test_model_availability():
            print("❌ Error: Model not available!")
            print("💡 Make sure you have:")
            print("  1. GPU with sufficient memory")
            print("  2. Hugging Face CLI logged in: huggingface-cli login")
            print("  3. Access to Llama models")
            sys.exit(1)
        
        if args.multiple_tokens:
            # Multi-token analysis mode - let model generate naturally
            print_model_header(args.model, args.text, args.raw_prompt)
            print(f"⏳ Generating natural completion and analyzing each step...")

            # Let model generate naturally without forcing token limits
            result = classifier.get_multi_token_logprobs(
                args.text,
                max_new_tokens=None,  # No limit - let model stop naturally
                raw_prompt=args.raw_prompt,
                chain_of_thought=args.chain_of_thought
            )

            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                sys.exit(1)

            # Show console output
            console_output = format_multi_token_console_output(result)
            print(console_output)

            # Generate markdown report
            markdown_content = generate_multi_token_markdown(result, args.model)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_suffix = "finetuned" if args.model == "finetuned" else "base"
            filename = f"logprob_analysis_{model_suffix}_{timestamp}.md"

            report_path = save_markdown_report(markdown_content, filename)
            print(f"\n📄 Markdown report saved to: {report_path}")

        elif args.quiet:
            # Simple output for scripting
            result = classifier.get_real_logprobs_confidence(args.text, raw_prompt=args.raw_prompt)
            if args.raw_prompt:
                # For raw prompts, just show the top predicted token
                token_details = result.get('token_details', {})
                if token_details:
                    top_token = list(token_details.keys())[0]
                    top_prob = list(token_details.values())[0]['prob']
                    print(f"{top_token} {top_prob:.4f}")
                else:
                    print("unknown 0.0000")
            else:
                prediction = result.get('prediction', 'unknown')
                confidence = result.get('confidence', 0.0)
                print(f"{prediction} {confidence:.4f}")
        else:
            # Full educational output
            print_model_header(args.model, args.text, args.raw_prompt)
            
            print("⏳ Analyzing token probabilities...")
            result = classifier.get_real_logprobs_confidence(args.text, raw_prompt=args.raw_prompt)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                sys.exit(1)
            
            # Show the detailed logprob table
            if args.raw_prompt:
                print("**When given this raw prompt, here's what the model computes for the next token:**")
            else:
                print("**When the model generates the first token of its response, here's what it actually computes:**")
            print()
            print("```")
            print(format_logprob_table(result, args.top_k))
            print("```")
            
            # Show which token was selected
            generated_token = result.get('generated_token', 'unknown')
            token_details = result.get('token_details', {})
            if generated_token in token_details:
                prob = token_details[generated_token]['prob'] * 100
                print(f"\n**The model selected: \"{generated_token}\" ({prob:.2f}% probability)**")
            
            # Show the aggregation summary only for classification prompts
            if not args.raw_prompt:
                print(format_aggregation_summary(result))
            
            # Add educational context
            if args.raw_prompt:
                print(f"\n🔍 **Raw Prompt Analysis**: This shows what tokens the model considers most likely")
                print(f"   to follow your input text. The log-probabilities reveal the model's internal")
                print(f"   uncertainty about what should come next.")
                
                # Show top token info
                if token_details:
                    top_token = list(token_details.keys())[0]
                    top_prob = list(token_details.values())[0]['prob']
                    if top_prob > 0.5:
                        print(f"   The model strongly predicts \"{top_token}\" as the next token ({top_prob:.1%}).")
                    else:
                        print(f"   The model is uncertain - top prediction \"{top_token}\" is only {top_prob:.1%}.")
            else:
                confidence = result.get('confidence', 0.0)
                if confidence > 0.9:
                    print(f"\n💡 **High Confidence**: The model is very certain about this classification.")
                    print(f"   The log-probabilities show a clear preference for one sentiment.")
                elif confidence > 0.7:
                    print(f"\n⚠️  **Medium Confidence**: The model has a preference but isn't completely certain.")
                else:
                    print(f"\n❓ **Low Confidence**: The model is genuinely uncertain about this text.")
                    print(f"   Notice how the probabilities are more evenly distributed across sentiment options.")
                
                print(f"\n🔍 **Technical Note**: Log-probabilities are negative values that the model")
                print(f"   actually computes. Higher probability = less negative log-prob.")
                print(f"   The confidence score ({confidence:.1%}) is mathematically derived from these probabilities.")
            
            # Universal technical note for raw prompts
            if args.raw_prompt:
                print(f"\n🔍 **Technical Note**: Log-probabilities are negative values that the model")
                print(f"   actually computes. Higher probability = less negative log-prob.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
