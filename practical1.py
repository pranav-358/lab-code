def smart_home_agent():
    while True:
        print("\n===== SMART HOME AI AGENT =====")
        print("1. Check Room Conditions")
        print("2. Exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            person = input("Is a person present in the room? (yes/no): ").lower()
            if person == "yes":
                print("Person detected in the room.")
                temperature = float(input("Enter temperature (°C): "))
                light = input("Is the room dark? (yes/no): ").lower()
                
                if temperature > 30:
                    print("Agent Action: Turn ON Fan")
                else:
                    print("Agent Action: Keep Fan OFF")
                
                if light == "yes":
                    print("Agent Action: Turn ON Light")
                elif light == "no":
                    print("Agent Action: Keep Light OFF")
                else:
                    print("Invalid light input! Enter yes or no.")
            elif person == "no":
                print("No person detected in the room.")
                print("Agent Action: Turn OFF Fan")
                print("Agent Action: Turn OFF Light")
            else:
                print("Invalid input! Enter yes or no.")
        elif choice == '2':
            print("AI Agent stopped.")
            break
        else:
            print("Invalid choice! Try again.")


smart_home_agent()
