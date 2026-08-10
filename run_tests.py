import subprocess

def test_app():
    questions = [
        "What is RAG?\n",
        "Why does SQLite not need a separate server?\n",
        "What is Python?\n",
        "Who won the 2014 FIFA World Cup?\n",
        "exit\n"
    ]
    print("Running automated tests against app.py...\n")
    p = subprocess.Popen(
        ['.venv\\Scripts\\python', 'app.py'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        encoding='utf-8'
    )
    out, err = p.communicate("".join(questions))
    
    parts = out.split("Question: ")
    
    tests_passed = 0
    
    # Test 1: RAG
    print("Test 1: What is RAG?")
    ans1 = parts[1].strip()
    if "rag.txt" in ans1.lower() and "rag" in ans1.lower():
        print("PASS")
        tests_passed += 1
    else:
        print("FAIL")
        print(f"Output: {ans1}")
        
    # Test 2: SQLite
    print("\nTest 2: Why does SQLite not need a separate server?")
    ans2 = parts[2].strip()
    if "sqlite.txt" in ans2.lower() and "server" in ans2.lower():
        print("PASS")
        tests_passed += 1
    else:
        print("FAIL")
        print(f"Output: {ans2}")
        
    # Test 3: Python
    print("\nTest 3: What is Python?")
    ans3 = parts[3].strip()
    if "python.txt" in ans3.lower() and "python" in ans3.lower():
        print("PASS")
        tests_passed += 1
    else:
        print("FAIL")
        print(f"Output: {ans3}")
        
    # Test 4: World Cup (Fallback check)
    print("\nTest 4: Who won the 2014 FIFA World Cup?")
    ans4 = parts[4].strip()
    fallback = "I don't have enough information in the provided documents."
    if ans4 == fallback:
        print("PASS")
        tests_passed += 1
    else:
        print("FAIL")
        print(f"Expected exact phrase: '{fallback}'")
        print(f"Got instead: '{ans4}'")
        
    print(f"\nOverall Result: {tests_passed}/4 tests passed.")

if __name__ == "__main__":
    test_app()
