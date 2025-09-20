import requests
import random

def test_confidence(text):
    try:
        response = requests.get(f'http://127.0.0.1:8000/classify?text={text}')
        if response.status_code == 200:
            result = response.json()
            return result.get('confidence', 0.0)
    except Exception as e:
        return None

def generate_truly_neutral_sports():
    # Completely neutral/contradictory templates
    templates = [
        "The sports situation presents both concerning and encouraging elements that cancel each other out.",
        "I observed equal amounts of positive and negative aspects in the athletic performance today.",
        "The team's effort contained simultaneously impressive and disappointing elements that neutralize overall assessment.",
        "This sports outcome defies clear categorization as either good or bad due to contradictory evidence.",
        "The athletic performance included both strengths and weaknesses that make evaluation genuinely impossible.",
        "I find myself unable to determine sentiment due to equally compelling positive and negative factors.",
        "The sports event contained paradoxical elements that prevent any definitive positive or negative judgment.",
        "This athletic situation presents contradictory indicators that make sentiment assessment genuinely unclear.",
        "The team performance exhibited both encouraging signs and troubling aspects in equal measure.",
        "I cannot form a sentiment opinion because positive elements are perfectly balanced by negative ones.",
        "The sports context contains mutually canceling positive and negative factors that defy classification.",
        "This athletic scenario presents opposing viewpoints that make sentiment determination logically impossible."
    ]
    return random.choice(templates)

def generate_truly_neutral_workplace():
    # Completely neutral/contradictory templates
    templates = [
        "The workplace situation presents both beneficial and problematic elements that cancel each other out.",
        "I observed equal amounts of positive and negative aspects in the professional environment today.",
        "The meeting contained simultaneously productive and unproductive elements that neutralize overall assessment.",
        "This work outcome defies clear categorization as either good or bad due to contradictory evidence.",
        "The professional interaction included both successes and failures that make evaluation genuinely impossible.",
        "I find myself unable to determine sentiment due to equally compelling favorable and unfavorable factors.",
        "The workplace event contained paradoxical elements that prevent any definitive positive or negative judgment.",
        "This professional situation presents contradictory indicators that make sentiment assessment genuinely unclear.",
        "The work environment exhibited both promising developments and concerning issues in equal measure.",
        "I cannot form a sentiment opinion because constructive elements are perfectly balanced by destructive ones.",
        "The workplace context contains mutually canceling beneficial and detrimental factors that defy classification.",
        "This professional scenario presents opposing viewpoints that make sentiment determination logically impossible."
    ]
    return random.choice(templates)

# Test the new generators extensively
print("Testing Truly Neutral Sports Examples:")
for i in range(10):
    example = generate_truly_neutral_sports()
    confidence = test_confidence(example)
    if confidence:
        print(f"  {confidence:.3f}: {example}")

print("\nTesting Truly Neutral Workplace Examples:")
for i in range(10):
    example = generate_truly_neutral_workplace()
    confidence = test_confidence(example)
    if confidence:
        print(f"  {confidence:.3f}: {example}")

# Find the best (lowest confidence) examples
print("\nFinding lowest confidence examples:")
best_sports = []
best_workplace = []

for _ in range(50):
    sports = generate_truly_neutral_sports()
    conf = test_confidence(sports)
    if conf and conf < 0.90:
        best_sports.append((conf, sports))

    workplace = generate_truly_neutral_workplace()
    conf = test_confidence(workplace)
    if conf and conf < 0.90:
        best_workplace.append((conf, workplace))

print(f"\nBest Sports Examples (confidence < 0.90):")
for conf, text in sorted(best_sports)[:5]:
    print(f"  {conf:.3f}: {text}")

print(f"\nBest Workplace Examples (confidence < 0.90):")
for conf, text in sorted(best_workplace)[:5]:
    print(f"  {conf:.3f}: {text}")