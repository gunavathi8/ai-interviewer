from agents.agent import InterviewerAgent
from workflow.graph import build_graph

def main():
    print("Welcome to the AI Interviewer Agent!")
    print("==================================")
    print("This tool simulates a technical interview with 3–5 questions.")
    
    agent = InterviewerAgent()
    graph = build_graph(agent)
    
    state = {}
    try:
        for output in graph.stream(state):
            pass  # Output handled in nodes
    except KeyboardInterrupt:
        print("\nInterview terminated by user.")
        exit(0)

if __name__ == "__main__":
    main()