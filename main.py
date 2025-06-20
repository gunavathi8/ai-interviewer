from agents.agent import InterviewerAgent
from workflow.graph import build_graph, InterviewState

def main():
    print("Welcome to the AI Interviewer Agent!")
    print("==================================")
    print("This tool simulates a technical interview with 3–5 questions.")
    
    agent = InterviewerAgent()
    graph = build_graph(agent)
    
    # Initialize state with default values
    state: InterviewState = {
        "topic": "",
        "question_count": 0,
        "questions": [],
        "answers": [],
        "scores": [],
        "feedbacks": [],
        "history": "",
        "current_question": {},
        "current_answer": "",
        "decision": ""
    }
    
    try:
        for output in graph.stream(state):
            print(f"Graph output: {output}")
    except KeyboardInterrupt:
        print("\nInterview terminated by user.")
        exit(0)

if __name__ == "__main__":
    main()