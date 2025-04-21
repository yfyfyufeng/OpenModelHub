import agent_main
import json
import os
import asyncio
from pathlib import Path
import sys
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"agent")])

async def test_json():
    # 使用相对于脚本的路径
    test_query_folder = current_dir / 'test_query'
    
    # 如果文件夹不存在，创建它
    if not test_query_folder.exists():
        test_query_folder.mkdir()
        print(f"Created test_query folder at {test_query_folder}")
        return

    json_files = [f for f in os.listdir(test_query_folder) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found in the test_query folder.")
        return

    print("Available JSON files:")
    for idx, file in enumerate(json_files, start=1):
        print(f"{idx}. {file}")

    file_choice = int(input("Select a file by number: ")) - 1

    if file_choice < 0 or file_choice >= len(json_files):
        print("Invalid choice.")
        return

    selected_file = test_query_folder / json_files[file_choice]
    print(f"Selected file: {selected_file}")
    
    with open(selected_file, 'r') as infile:
        tests = json.load(infile)
        num_tests = len(tests)
        print(f"Number of test_cases: {num_tests}")
        for i in range(num_tests):
            test = tests[i]
            query = test['query']

            print(f"Test {i+1}, input: {query}.")
            print('output: ')
            await agent_main.query_agent(query)
            print("--------------------------------------------------")

if __name__ == "__main__":
    print("Agent test started")
    while True:
        print("Available tests:")
        print("1. Test with json input;")
        choice = input("Enter your choice; x to exit.\n> ")
        if choice == '1':
            print("Running test with json input...")
            asyncio.run(test_json())
        elif choice == 'x':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
