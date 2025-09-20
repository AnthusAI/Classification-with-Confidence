# Dataset - Expanded Flat File Format

This directory contains the expanded sentiment dataset in a simple flat file format for easy management and generation.

## File Structure

Each file contains examples for one sentiment category:

- `strong_positive.txt` - Clearly positive sentiment examples
- `strong_negative.txt` - Clearly negative sentiment examples  
- `medium_positive.txt` - Moderately positive sentiment examples
- `medium_negative.txt` - Moderately negative sentiment examples
- `weak_positive.txt` - Mildly positive sentiment examples
- `weak_negative.txt` - Mildly negative sentiment examples
- `neutral_positive.txt` - Truly neutral examples (sports context, labeled positive for domain bias)
- `neutral_negative.txt` - Truly neutral examples (workplace context, labeled negative for domain bias)

## Format

- **One example per line** (no quotes needed)
- **Comments start with #** and are ignored
- **Empty lines are ignored**
- **Simple text only** - no JSON or special formatting

## Usage for Coding Agents

### CLI Classifier Tool

Use the CLI classifier to test examples before adding them:

```bash
# Test a potential example
python classify_cli.py "Your example text here"

# Get simple output for scripting
python classify_cli.py --quiet "Your example text here"
# Output: positive N/A
```

### Adding Examples

1. **Test your example** with the CLI classifier first
2. **Add to appropriate file** based on intended sentiment and strength
3. **One line per example** - just paste the text

Example workflow:
```bash
# Test an example
python classify_cli.py "This movie was surprisingly decent"
# Output shows: positive

# Add to weak_positive.txt since it's mildly positive
echo "This movie was surprisingly decent" >> dataset/weak_positive.txt
```

### Loading the Dataset

Use the dataset loader in Python:

```python
from dataset_loader import DatasetLoader

loader = DatasetLoader()
loader.print_summary()  # Show stats

all_examples = loader.load_all()  # Get all examples
```

## Goals

- **Scale up to 10,000+ examples** (order of magnitude increase)
- **Create more challenging examples** that aren't trivially easy for Llama 3.1
- **Balanced distribution** across all sentiment categories
- **Real-world complexity** - avoid obvious patterns

## Guidelines for Example Generation

### Strong Positive/Negative
- Clear, unambiguous sentiment
- No contradictory elements
- Examples: "Absolutely love this!" / "Completely terrible!"

### Medium Positive/Negative  
- Clear sentiment but less intense
- Examples: "Pretty good overall" / "Not very impressive"

### Weak Positive/Negative
- Subtle sentiment, could be borderline
- Examples: "It's okay I guess" / "Could be better"

### Neutral Positive/Negative (NEW)
- **Truly neutral sentiment** - no positive or negative indicators
- **Domain-specific context** for fine-tuning bias learning
- **Neutral Positive**: Sports context, factual statements
- **Neutral Negative**: Workplace context, factual statements
- **Goal**: Create very low confidence examples for base model
- Examples: "The game started at 7 PM" / "The meeting was scheduled for Tuesday"

### Avoid These Patterns
- **Obvious sarcasm** ("Great, another meeting" → negative)
- **Contradictory sports/news patterns** (too learnable)
- **Formulaic structures** that create obvious patterns

### Create These Challenges
- **Genuine ambiguity** where context matters
- **Subtle emotional cues** requiring deeper understanding
- **Cultural references** that need broader knowledge
- **Complex sentences** with multiple sentiment elements
