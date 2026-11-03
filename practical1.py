"""
PR1: INTELLIGENT AGENTS AND PEAS REPRESENTATION (CO1)
Application: Smart Home Controller

This program demonstrates a simple rule-based intelligent agent.
It receives sensor readings, makes decisions using condition-action rules,
and controls smart-home devices.

Run:
    python PR1_Smart_Home_Controller.py
"""


# ============================================================
# 1. PEAS REPRESENTATION
# ============================================================

def display_peas():
    """Display the PEAS description of the Smart Home Controller."""
    print("\n" + "=" * 60)
    print("          PEAS REPRESENTATION - SMART HOME CONTROLLER")
    print("=" * 60)

    print("\nP - PERFORMANCE MEASURE")
    print("  1. Maintain home security.")
    print("  2. Provide comfort through temperature and lighting control.")
    print("  3. Detect emergencies such as smoke.")
    print("  4. Reduce unnecessary energy consumption.")

    print("\nE - ENVIRONMENT")
    print("  House rooms, doors, windows, occupants, temperature,")
    print("  lighting conditions and possible emergency situations.")

    print("\nA - ACTUATORS")
    print("  Door lock, security camera, alarm, fan, lights and curtains.")

    print("\nS - SENSORS")
    print("  Motion sensor, temperature sensor, smoke sensor,")
    print("  light sensor and door sensor.")

    print("=" * 60)


# ============================================================
# 2. INITIAL SENSOR AND DEVICE STATES
# ============================================================

# Sensor readings
motion_detected = False
temperature = 25.0
smoke_detected = False
light_level = 50
door_open = False

# Smart-home device states
door_lock = "LOCKED"
security_camera = "OFF"
fire_alarm = "OFF"
fan = "OFF"
lights = "OFF"
curtains = "CLOSED"

# Store actions performed by the agent
action_log = []


# ============================================================
# 3. INPUT VALIDATION FUNCTIONS
# ============================================================

def get_yes_no(prompt):
    """Accept only yes/no input and return True or False."""
    while True:
        answer = input(prompt).strip().lower()

        if answer in ("yes", "y"):
            return True
        elif answer in ("no", "n"):
            return False
        else:
            print("Invalid input. Please enter yes or no.")


def get_number(prompt, minimum=None, maximum=None):
    """Accept and validate a numeric input."""
    while True:
        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Enter a value greater than or equal to {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Enter a value less than or equal to {maximum}.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


# ============================================================
# 4. UPDATE SENSOR READINGS
# ============================================================

def update_sensors():
    """Read values from the simulated smart-home sensors."""
    global motion_detected, temperature
    global smoke_detected, light_level, door_open

    print("\n" + "-" * 50)
    print("             UPDATE SENSOR READINGS")
    print("-" * 50)

    motion_detected = get_yes_no("Is motion detected? (yes/no): ")
    temperature = get_number("Enter room temperature in Celsius: ")
    smoke_detected = get_yes_no("Is smoke detected? (yes/no): ")
    light_level = get_number("Enter light level (0-100): ", 0, 100)
    door_open = get_yes_no("Is the main door open? (yes/no): ")

    print("\nSensor readings updated successfully.")


# ============================================================
# 5. INTELLIGENT AGENT - DECISION MAKING
# ============================================================

def run_intelligent_agent():
    """
    Apply condition-action rules to sensor readings.
    The agent observes the environment, decides what to do,
    and operates the smart-home devices.
    """
    global door_lock, security_camera, fire_alarm
    global fan, lights, curtains

    print("\n" + "=" * 50)
    print("          INTELLIGENT AGENT IS WORKING")
    print("=" * 50)

    # Rule 1: Smoke detection has the highest priority.
    if smoke_detected:
        fire_alarm = "ON"
        security_camera = "ON"
        fan = "OFF"
        print("ALERT: Smoke detected. Fire alarm activated.")
        action_log.append("Smoke detected: fire alarm and camera turned ON.")
    else:
        fire_alarm = "OFF"
        print("No smoke detected. Fire alarm is OFF.")

    # Rule 2: Lock the door when it is closed.
    if not door_open:
        door_lock = "LOCKED"
        print("Main door is closed. Door lock is LOCKED.")
    else:
        door_lock = "UNLOCKED"
        print("Main door is open. Door lock remains UNLOCKED.")

    # Rule 3: Activate the security camera when motion is detected.
    if motion_detected:
        security_camera = "ON"
        print("Motion detected. Security camera turned ON.")
        action_log.append("Motion detected: security camera turned ON.")
    elif not smoke_detected:
        security_camera = "OFF"
        print("No motion detected. Security camera turned OFF.")

    # Rule 4: Turn on the fan when temperature is above 30°C.
    # Smoke emergency takes priority, so the fan remains OFF during smoke.
    if temperature > 30 and not smoke_detected:
        fan = "ON"
        print("High temperature detected. Fan turned ON.")
        action_log.append("High temperature: fan turned ON.")
    else:
        fan = "OFF"
        if not smoke_detected:
            print("Temperature is comfortable. Fan turned OFF.")

    # Rule 5: Turn on lights when the room is dark.
    if light_level < 30:
        lights = "ON"
        print("Low light detected. Lights turned ON.")
        action_log.append("Low light: lights turned ON.")
    else:
        lights = "OFF"
        print("Sufficient light detected. Lights turned OFF.")

    # Rule 6: Open curtains when there is sufficient daylight.
    if light_level >= 60:
        curtains = "OPEN"
        print("Sufficient daylight. Curtains opened.")
    else:
        curtains = "CLOSED"
        print("Low daylight. Curtains closed.")

    action_log.append("Intelligent agent completed one decision cycle.")

    print("\nAgent decision cycle completed.")
    print("=" * 50)


# ============================================================
# 6. SMART HOME STATUS
# ============================================================

def display_status():
    """Display current sensor readings and device states."""
    print("\n" + "=" * 55)
    print("                 SMART HOME STATUS")
    print("=" * 55)

    print("\nSENSOR READINGS")
    print("-" * 55)
    print(f"Motion Detected : {'YES' if motion_detected else 'NO'}")
    print(f"Temperature     : {temperature:.1f} °C")
    print(f"Smoke Detected  : {'YES' if smoke_detected else 'NO'}")
    print(f"Light Level     : {light_level:.0f}/100")
    print(f"Main Door Open  : {'YES' if door_open else 'NO'}")

    print("\nDEVICE STATUS")
    print("-" * 55)
    print(f"Door Lock       : {door_lock}")
    print(f"Security Camera : {security_camera}")
    print(f"Fire Alarm      : {fire_alarm}")
    print(f"Fan             : {fan}")
    print(f"Lights          : {lights}")
    print(f"Curtains        : {curtains}")

    print("=" * 55)


# ============================================================
# 7. VIEW AGENT ACTION LOG
# ============================================================

def display_action_log():
    """Display actions performed by the intelligent agent."""
    print("\n" + "=" * 55)
    print("                  AGENT ACTION LOG")
    print("=" * 55)

    if not action_log:
        print("No agent actions recorded yet.")
    else:
        for count, action in enumerate(action_log, start=1):
            print(f"{count}. {action}")

    print("=" * 55)


# ============================================================
# 8. MAIN MENU
# ============================================================

def main():
    """Run the Smart Home Controller application."""
    print("\n" + "=" * 55)
    print("             SMART HOME CONTROLLER")
    print("        Rule-Based Intelligent Agent")
    print("=" * 55)

    display_peas()

    while True:
        print("\n--------------- MAIN MENU ---------------")
        print("1. Display PEAS Representation")
        print("2. Update Sensor Readings")
        print("3. Run Intelligent Agent")
        print("4. Display Smart Home Status")
        print("5. View Agent Action Log")
        print("6. Exit")
        print("-----------------------------------------")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            display_peas()

        elif choice == "2":
            update_sensors()

        elif choice == "3":
            run_intelligent_agent()

        elif choice == "4":
            display_status()

        elif choice == "5":
            display_action_log()

        elif choice == "6":
            print("\nSmart Home Controller closed successfully.")
            print("Thank you!")
            break

        else:
            print("\nInvalid choice. Please enter a number from 1 to 6.")


# ============================================================
# 9. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
