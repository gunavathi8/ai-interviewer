from langgraph.graph import StateGraph, END
from agents.agent import InterviewerAgent
from utils.types import InterviewState

def build_graph(agent: InterviewerAgent):
    workflow = StateGraph(InterviewState)

    workflow.add_node("select_topic", agent.select_topic)
    workflow.add_node("generate_question", agent.generate_question)
    workflow.add_node("collect_answer", agent.collect_answer)
    workflow.add_node("evaluate_answer", agent.evaluate_answer)
    workflow.add_node("decide_next", decide_next)
    workflow.add_node("generate_feedback", agent.generate_feedback)

    workflow.set_entry_point("select_topic")
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

def decide_next(state: InterviewState) -> InterviewState:
    print(f"Deciding next step. State: {state}")
    if state.get("decision") == "end" or state.get("question_count", 0) >= 5:
        state["decision"] = "end"
    elif state.get("scores") and state["scores"][-1] < 5:
        state["decision"] = "continue"
    else:
        state["decision"] = "continue"
    return state