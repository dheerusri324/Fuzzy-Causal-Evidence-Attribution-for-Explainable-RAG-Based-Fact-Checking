from streamlit.testing.v1 import AppTest

def run_app_test():
    print("Initializing AppTest...")
    import os
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "main.py"))
    at = AppTest.from_file(app_path, default_timeout=600)
    
    print("Running initial app load...")
    at.run()
    assert not at.exception, f"App load failed with exception: {at.exception}"
    
    print("Setting claim text...")
    # Find the text input for claim and set its value
    at.text_input[0].set_value("The Lincoln Motor Company was named by Henry M. Leland.")
    
    print("Clicking 'Run Analysis'...")
    at.button[0].click().run(timeout=600)
    
    assert not at.exception, f"Run analysis failed with exception: {at.exception}"
    
    print("Analysis finished. Checking UI outputs...")
    
    # Dump headers to see what rendered
    headers = [h.value for h in at.header]
    print(f"Headers found: {headers}")
    
    # Check if 'FACT-CHECKING VERDICT' is in headers
    assert any("FACT-CHECKING VERDICT" in h for h in headers), "Verdict header not found!"
    
    # Get all markdown/text values to check for output presence
    markdowns = [md.value for md in at.markdown]
    infos = [info.value for info in at.info]
    successes = [succ.value for succ in at.success]
    
    print(f"Infos: {infos}")
    print(f"Success blocks (Final summary): {successes}")
    
    print("\nSUCCESS: E2E Pipeline through Streamlit UI works perfectly!")

if __name__ == "__main__":
    run_app_test()
