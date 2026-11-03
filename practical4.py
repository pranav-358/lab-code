"""
PR4: Knowledge Representation and Expert System
Rule-Based Expert System using Forward Chaining and Backward Chaining

This is a simple educational demonstration of propositional logic.
The symptom rules are illustrative only and are NOT a medical diagnosis.
"""

# -------------------- KNOWLEDGE BASE --------------------
# Each rule has the form:
# IF all conditions are true, THEN the conclusion is true.
#
# Example:
# fever AND cough -> respiratory_infection

RULES = [
    (("fever", "cough"), "respiratory_infection"),
    (("fever", "headache"), "viral_illness"),
    (("sneezing", "runny_nose"), "allergy_or_cold"),
    (("respiratory_infection",), "medical_consultation"),
    (("viral_illness",), "rest_and_hydration"),
    (("allergy_or_cold",), "monitor_symptoms"),
]

SYMPTOMS = ["fever", "cough", "headache", "sneezing", "runny_nose"]

# -------------------- FORWARD CHAINING --------------------
def forward_chaining(initial_facts):
    """Derive all conclusions that follow from the known facts."""
    facts = set(initial_facts)
    derived = []

    changed = True
    while changed:
        changed = False

        for conditions, conclusion in RULES:
            if all(condition in facts for condition in conditions):
                if conclusion not in facts:
                    facts.add(conclusion)
                    derived.append(conclusion)
                    changed = True
                    print("Derived:", conclusion.replace("_", " "))

    return facts, derived


# -------------------- BACKWARD CHAINING --------------------
def backward_chaining(goal, facts, active_goals=None):
    """Check whether a goal can be proved from facts and rules."""
    if goal in facts:
        return True

    if active_goals is None:
        active_goals = set()

    # Prevent infinite recursion if rules contain a cycle.
    if goal in active_goals:
        return False

    # Find a rule whose conclusion matches the requested goal.
    matching_rules = [
        conditions for conditions, conclusion in RULES
        if conclusion == goal
    ]

    if not matching_rules:
        return False

    next_active = active_goals | {goal}

    # A goal is true if at least one matching rule has all conditions true.
    for conditions in matching_rules:
        if all(
            backward_chaining(condition, facts, next_active)
            for condition in conditions
        ):
            return True

    return False


# -------------------- HELPER FUNCTIONS --------------------
def get_yes_no(prompt):
    """Accept only yes or no as input."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("yes", "y"):
            return True
        if answer in ("no", "n"):
            return False
        print("Please enter yes or no.")


def show_rules():
    print("\nKnowledge Base (Rules):")
    for number, (conditions, conclusion) in enumerate(RULES, start=1):
        left_side = " AND ".join(item.replace("_", " ") for item in conditions)
        right_side = conclusion.replace("_", " ")
        print(f"R{number}: IF {left_side} THEN {right_side}")


# -------------------- MAIN PROGRAM --------------------
def main():
    print("=" * 55)
    print("       RULE-BASED EXPERT SYSTEM")
    print("       Forward and Backward Chaining")
    print("=" * 55)
    print("\nEnter the symptoms based on the questions below.")
    print("This is an academic rule-based demonstration, not medical advice.")

    facts = set()
    for symptom in SYMPTOMS:
        if get_yes_no(f"Do you have {symptom.replace('_', ' ')}? (yes/no): "):
            facts.add(symptom)

    print("\nInitial Facts:")
    if facts:
        for fact in sorted(facts):
            print("-", fact.replace("_", " "))
    else:
        print("No symptoms were entered.")

    show_rules()

    print("\n" + "=" * 55)
    print("FORWARD CHAINING")
    print("=" * 55)
    final_facts, derived = forward_chaining(facts)

    print("\nFinal Knowledge Base:")
    for fact in sorted(final_facts):
        print("-", fact.replace("_", " "))

    if not derived:
        print("No new conclusions could be derived.")

    print("\n" + "=" * 55)
    print("BACKWARD CHAINING")
    print("=" * 55)
    print("Available goals:")
    goals = sorted({conclusion for _, conclusion in RULES})
    for item in goals:
        print("-", item)

    goal = input("\nEnter a goal to check: ").strip().lower().replace(" ", "_")
    if backward_chaining(goal, facts):
        print("Goal established:", goal.replace("_", " "))
    else:
        print("Goal could not be established:", goal.replace("_", " "))


if __name__ == "__main__":
    main()
