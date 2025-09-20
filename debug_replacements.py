import requests
import random

def test_confidence(text):
    try:
        response = requests.get(f'http://127.0.0.1:8000/classify?text={text}')
        if response.status_code == 200:
            result = response.json()
            return result.get('confidence', 0.0)
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_ultra_weak_positive_sports():
    # Much more ambiguous templates
    templates = [
        "I'm genuinely unsure whether the sports activity was marginally positive or just average overall.",
        "The athletic performance left me with mixed feelings about whether it represents any meaningful progress.",
        "I can't quite determine if the team's effort today was somewhat encouraging or simply routine.",
        "It's unclear to me whether the sports outcome suggests mild improvement or continued mediocrity.",
        "I have conflicted thoughts about whether the athletic display was moderately inspiring or unremarkable.",
        "The sports session was difficult to interpret - possibly slightly positive but maybe just standard.",
        "I'm uncertain if the team's performance indicates marginal progress or persistent stagnation.",
        "The athletic effort seemed ambiguous - perhaps minimally encouraging but hard to assess definitively."
    ]
    return random.choice(templates)

def generate_ultra_weak_negative_workplace():
    # Much more ambiguous templates
    templates = [
        "I'm genuinely uncertain whether the workplace situation was mildly concerning or just typical.",
        "The professional environment left me with mixed impressions about whether issues are emerging.",
        "I can't quite determine if the meeting outcomes were somewhat problematic or simply routine.",
        "It's unclear to me whether the work dynamics suggest minor concerns or normal variations.",
        "I have conflicted thoughts about whether the project status was moderately troubling or standard.",
        "The workplace interaction was difficult to interpret - possibly slightly negative but maybe normal.",
        "I'm uncertain if the professional feedback indicates minor problems or typical workplace dynamics.",
        "The work situation seemed ambiguous - perhaps minimally concerning but hard to assess definitively."
    ]
    return random.choice(templates)

# Test both generators
print("Testing Ultra Weak Positive Sports Examples:")
for i in range(5):
    example = generate_ultra_weak_positive_sports()
    confidence = test_confidence(example)
    print(f"  {confidence:.3f}: {example}")

print("\nTesting Ultra Weak Negative Workplace Examples:")
for i in range(5):
    example = generate_ultra_weak_negative_workplace()
    confidence = test_confidence(example)
    print(f"  {confidence:.3f}: {example}")

# Test some existing examples for comparison
print("\nTesting some existing examples for comparison:")
existing_examples = [
    "The team played really well today",
    "Our meeting was very productive",
    "I suppose the practice was somewhat better",
    "Perhaps the workplace dynamics were moderately concerning"
]

for example in existing_examples:
    confidence = test_confidence(example)
    print(f"  {confidence:.3f}: {example}")