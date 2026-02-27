import os
from dotenv import load_dotenv

# import the local graph module via the engine package
from engine.graph import research_app, AgentState  # type: ignore

def run_research():
    print("InsightEngine Pro Activate")
    question = input("What would you like to research from your PDFs?")

    # initialize state
    initial_state: AgentState = {
        "question": question,
        "context": [],
        "answer": "",
        "steps": []
    }

    # execute graph
    final_output = research_app.invoke(initial_state)

    print("\n" + "="*30)
    print("FINAL ANSWER:")
    print(final_output["answer"])
    print("="*30)
    print(f"Workflow Path: {' -> '.join(final_output['steps'])}")

if __name__ == "__main__":
    run_research()