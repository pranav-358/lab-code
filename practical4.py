
# PR4: Rule-Based Expert System
# Forward Chaining and Backward Chaining

# Knowledge Base: Rules
rules = [
    (["fever", "cough"], "respiratory_infection"),
    (["fever", "headache"], "viral_illness"),
    (["sneezing", "runny_nose"], "allergy_or_cold"),
    (["respiratory_infection"], "medical_consultation"),
    (["viral_illness"], "rest_and_hydration"),
    (["allergy_or_cold"], "monitor_symptoms")
]


# Forward Chaining
def forward_chaining(facts):
    facts = set(facts)
    changed = True

    while changed:
        changed = False

        for conditions, conclusion in rules:
            if all(condition in facts for condition in conditions):
                if conclusion not in facts:
                    facts.add(conclusion)
                    changed = True
                    print("Derived:", conclusion)

    return facts


# Backward Chaining
def backward_chaining(goal, facts, visited=None):
    if visited is None:
        visited = set()

    if goal in facts:
        return True

    if goal in visited:
        return False

    visited.add(goal)

    for conditions, conclusion in rules:
        if conclusion == goal:
            if all(
                backward_chaining(condition, facts, visited.copy())
                for condition in conditions
            ):
                return True

    return False


# Main Program
print("===== Rule-Based Expert System =====")

print("\nEnter symptoms (yes/no):")

symptoms = [
    "fever",
    "cough",
    "headache",
    "sneezing",
    "runny_nose"
]

facts = []

for symptom in symptoms:
    answer = input(f"Do you have {symptom.replace('_', ' ')}? ").strip().lower()

    if answer == "yes":
        facts.append(symptom)


# Forward Chaining
print("\n===== Forward Chaining =====")

derived_facts = forward_chaining(facts)

print("\nFinal Knowledge Base:")
for fact in sorted(derived_facts):
    print("-", fact)


# Backward Chaining
print("\n===== Backward Chaining =====")

goal = input(
    "\nEnter a goal to check "
    "(e.g., medical_consultation, rest_and_hydration): "
).strip().lower()

if backward_chaining(goal, set(facts)):
    print("Goal established:", goal)
else:
    print("Goal could not be established:", goal)
