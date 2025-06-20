from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from agents.agent import InterviewerAgent

class InterviewState(TypedDict):
    topic: str
    question_count: int
    questions: List[dict]
    answers: List[str]
    scores: List[int]
    feedbacks: List[str]
    history: str
    current_question: dict
    current_answer: str
    decision: str


def build_graph(agent: InterviewerAgent):
    workflow = StateGraph(InterviewState)

    workflow.add_node("select_topic", agent.select_topic)
    workflow.add_node("generate_question", agent.generate_question)
    workflow.add_node("collect_answer", agent.collect_answer)
    workflow.add_node("evaluate_answer", agent.evaluate_answer)
    workflow.add_node("decide_next", lambda state: {"decision": decide_next(state)})
    workflow.add_node("generate_feedback", agent.generate_feedback)
    
    workflow.set_entry_point("select_topic")
    workflow.add_edge(START, "select_topic")
    workflow.add_edge("select_topic", "generate_question")
    workflow.add_edge("generate_question", "collect_answer")
    workflow.add_edge("collect_answer", "evaluate_answer")
    workflow.add_edge("evaluate_answer", "decide_next")
    workflow.add_conditional_edges(
        "decide_next",
        lambda state: state["decision"],
        {
            "continue": "generate_question",
            "end": "generate_feedback"
        }
    )
    workflow.add_edge("generate_feedback", END)
    
    return workflow.compile()

def decide_next(state: InterviewState):
    if state["question_count"] < 5 and state["scores"][-1] < 5:
        return "continue"
    elif state["question_count"] >= 5:
        return "end"
    return "continue"

