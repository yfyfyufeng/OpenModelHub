import agent_main
import json

def test_json():
    
    choice = input("Do you want to print result to console, too? y/n")
    
    to_console = choice.lower() == 'y'
    
    with open('input.json', 'r') as infile:
        with open('dump.txt', 'w') as outfile:
            
            tests = json.load(infile)
            num_tests = len(query)
            print(f"Number of test_cases: {num_tests}")
            for i in range(num_tests):
                test = tests[i]
                query = test['query']
                result = agent_main.main(query)
                outfile.write(f"Test {i+1}, input: {query}.\n")
                outfile.write(result)
                outfile.write("--------------------------------------------------\n")
                if to_console:
                    print(f"Test {i+1}, input: {query}.")
                    print(f'output:\n{result}')
                    print("--------------------------------------------------")

if __name__ == "__main__":
    print("Agent test started")
    while True:
        print("Available tests:")
        print("1. Test with json input;")
        choice = input("Enter your choice; x to exit.")
        if choice == '1':
            print("Running test with json input...")
            # agent_main.run_agent()  # Uncomment this line to run the agent
        elif choice == 'x':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
