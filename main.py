from agents.agent import InterviewerAgent
from workflow.graph import build_graph, InterviewState

def main():
    print("Welcome to the AI Interviewer Agent!")
    print("==================================")
    print("This tool simulates a technical interview with 5 questions.")
    
    agent = InterviewerAgent()
    graph = build_graph(agent)
    
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
        "decision": "",
        "current_difficulty": "easy",
        "is_follow_up": False,
        "hint_count": 0
    }
    
    try:
        for _ in graph.stream(state, config={"recursion_limit": 50}):
            pass
    except KeyboardInterrupt:
        print("\nInterview terminated by user.")
        exit(0)

if __name__ == "__main__":
    main()