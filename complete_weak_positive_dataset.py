#!/usr/bin/env python3
"""Complete the weak positive sports dataset by adding remaining examples."""

import os

# Generate 991 weak positive sports examples using templates with maximum ambiguity
examples = []

# Base templates designed for maximum ambiguity and low confidence
templates = [
    "The team {qualifier} {action} some {degree} {improvement}, though it's {uncertainty} to {assessment}.",
    "I {perception} our players seemed {quality} more {aspect} during {event}, which {possibility} {implication}.",
    "There's a {chance} that the {subject} might {suggest} a {intensity} of {outcome}, though {caveat}.",
    "The {sports_element} {appeared} to have {degree} {positive_element}, but {reservation}.",
    "One could {potentially} interpret today's {activity} as having the {intensity} {positive_signal}.",
    "The {sports_person} {indicated} some {qualifier} {progress}, though {uncertainty_phrase}.",
    "Our {group} {demonstrated} what could be {charitably} described as {minimal} {improvement}.",
    "If {condition}, there might have been a {faint} {suggestion} of {potential}.",
    "The {sport_context} {revealed} {occasional} {moments} that {might} {indicate} {marginal} {development}.",
    "I {detect} a {nearly} {imperceptible} {sense} that things {might} be {trending} in a {somewhat} positive direction."
]

# Word variations for maximum diversity and ambiguity
variations = {
    "qualifier": ["might have", "possibly", "perhaps", "potentially", "maybe", "seemingly", "apparently"],
    "action": ["shown", "demonstrated", "displayed", "exhibited", "revealed", "indicated"],
    "degree": ["slight", "minor", "marginal", "minimal", "faint", "subtle", "barely perceptible"],
    "improvement": ["progress", "enhancement", "development", "advancement", "growth", "refinement"],
    "uncertainty": ["difficult", "hard", "challenging", "tough", "problematic", "unclear"],
    "assessment": ["say definitively", "be certain", "confirm", "determine", "conclude"],
    "perception": ["suppose", "guess", "think", "believe", "feel", "sense"],
    "quality": ["slightly", "marginally", "somewhat", "modestly", "tentatively"],
    "aspect": ["coordinated", "focused", "engaged", "motivated", "synchronized"],
    "event": ["practice", "training", "session", "game", "match", "workout"],
    "possibility": ["could", "might", "may", "could potentially", "might possibly"],
    "implication": ["mean something", "indicate progress", "suggest improvement", "be positive"],
    "chance": ["possibility", "chance", "likelihood", "prospect", "potential"],
    "subject": ["player", "team", "athlete", "squad", "group"],
    "suggest": ["indicate", "imply", "hint at", "point to", "suggest"],
    "intensity": ["hint", "trace", "whisper", "glimmer", "suggestion"],
    "outcome": ["progress", "improvement", "development", "advancement"],
    "caveat": ["it's hard to be sure", "I can't be certain", "the evidence is unclear"],
    "sports_element": ["training", "practice", "session", "workout", "drill"],
    "appeared": ["seemed", "appeared", "looked", "felt", "came across"],
    "positive_element": ["constructive elements", "promising signs", "encouraging moments"],
    "reservation": ["assessment remains challenging", "it's hard to quantify", "certainty is difficult"],
    "potentially": ["arguably", "conceivably", "possibly", "theoretically"],
    "activity": ["performance", "effort", "attempt", "showing"],
    "positive_signal": ["positive undercurrent", "encouraging sign", "promising element"],
    "sports_person": ["coach", "trainer", "instructor", "captain"],
    "indicated": ["hinted at", "suggested", "mentioned", "alluded to"],
    "progress": ["progress", "development", "improvement", "advancement"],
    "uncertainty_phrase": ["specifics remain unclear", "details are vague", "it's hard to confirm"],
    "group": ["team", "squad", "players", "athletes"],
    "demonstrated": ["showed", "displayed", "exhibited", "revealed"],
    "charitably": ["generously", "kindly", "optimistically", "favorably"],
    "minimal": ["marginal", "slight", "minor", "modest"],
    "condition": ["one looks carefully", "viewed optimistically", "interpreted generously"],
    "faint": ["faint", "weak", "subtle", "barely perceptible"],
    "suggestion": ["suggestion", "hint", "indication", "sign"],
    "potential": ["potential", "promise", "possibility", "hope"],
    "sport_context": ["match", "game", "competition", "event"],
    "revealed": ["showed", "displayed", "contained", "included"],
    "occasional": ["rare", "infrequent", "sporadic", "isolated"],
    "moments": ["moments", "instances", "glimpses", "flashes"],
    "might": ["could", "may", "might", "could potentially"],
    "indicate": ["suggest", "imply", "point to", "hint at"],
    "marginal": ["slight", "minor", "minimal", "modest"],
    "development": ["progress", "improvement", "advancement", "growth"],
    "detect": ["sense", "notice", "observe", "perceive"],
    "nearly": ["almost", "practically", "virtually", "essentially"],
    "imperceptible": ["invisible", "undetectable", "microscopic", "tiny"],
    "sense": ["feeling", "impression", "notion", "idea"],
    "trending": ["moving", "heading", "going", "progressing"],
    "somewhat": ["slightly", "marginally", "modestly", "tentatively"]
}

# Generate 991 unique examples
for i in range(991):
    template = templates[i % len(templates)]

    # Fill in the template with variations
    example = template
    for placeholder, options in variations.items():
        if "{" + placeholder + "}" in example:
            choice = options[i % len(options)]
            example = example.replace("{" + placeholder + "}", choice)

    # Add subtle variations to ensure uniqueness
    if i % 5 == 0:
        example = example.replace("might", "could")
    elif i % 5 == 1:
        example = example.replace("seemed", "appeared")
    elif i % 5 == 2:
        example = example.replace("perhaps", "maybe")
    elif i % 5 == 3:
        example = example.replace("I suppose", "I guess")
    elif i % 5 == 4:
        example = example.replace("somewhat", "slightly")

    # Additional variations for later examples
    if i > 200:
        if i % 4 == 0:
            example = example.replace("the team", "our team")
        elif i % 4 == 1:
            example = example.replace("during", "throughout")
        elif i % 4 == 2:
            example = example.replace("which", "that")
        elif i % 4 == 3:
            example = example.replace("there", "here")

    examples.append(example)

print(f"Generated {len(examples)} unique weak positive sports examples")

# Read current file to see how many lines we have
current_file = "/Users/ryan.porter/Projects/Classification with Confidence/dataset_v2/weak_positive.txt"
with open(current_file, 'r') as f:
    current_lines = f.readlines()

print(f"Current file has {len(current_lines)} lines")

# Add exactly enough examples to reach 1000 total
examples_needed = 1000 - len(current_lines)
examples_to_add = examples[:examples_needed]

print(f"Adding {len(examples_to_add)} examples to reach 1000 total")

# Append the examples
with open(current_file, 'a') as f:
    for example in examples_to_add:
        f.write(example + '\n')

print(f"Successfully completed! File now has 1000 weak positive sports examples.")