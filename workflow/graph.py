from langgraph.graph import StateGraph, END
from agents.agent import InterviewerAgent
from utils.types import InterviewState
import logging

class InterviewState(InterviewState):
    current_difficulty: str
    is_follow_up: bool
    hint_count: int

def build_graph(agent: InterviewerAgent):
    workflow = StateGraph(InterviewState)

    workflow.add_node("select_topic", agent.select_topic)
    workflow.add_node("generate_question", agent.generate_question)
    workflow.add_node("collect_answer", agent.collect_answer)
    workflow.add_node("evaluate_answer", agent.evaluate_answer)
    workflow.add_node("generate_hint", agent.generate_hint)
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
            "hint": "generate_hint",
            "end": "generate_feedback"
        }
    )
    workflow.add_edge("generate_hint", "collect_answer")
    workflow.add_edge("generate_feedback", END)

    return workflow.compile()

def decide_next(state: InterviewState) -> InterviewState:
    difficulty_levels = ["easy", "medium", "hard"]
    current_difficulty = state.get("current_difficulty", "easy")
    current_index = difficulty_levels.index(current_difficulty)

    if state.get("decision") == "end" or state.get("question_count", 0) >= 5:
        state["decision"] = "end"
        state["is_follow_up"] = False
        logging.info(f"Decision: end (question_count={state['question_count']})")
        return state

    last_score = state["scores"][-1] if state["scores"] else 0
    if last_score < 4 and state.get("hint_count", 0) < 1:
        state["decision"] = "hint"
        state["current_difficulty"] = difficulty_levels[max(0, current_index - 1)]
        logging.info(f"Decision: hint (score={last_score}, hint_count={state['hint_count']}, new_difficulty={state['current_difficulty']})")
    else:
        state["decision"] = "continue"
        state["is_follow_up"] = False
        if last_score >= 7:
            state["current_difficulty"] = difficulty_levels[min(len(difficulty_levels) - 1, current_index + 1)]
        elif last_score < 4:
            state["current_difficulty"] = difficulty_levels[max(0, current_index - 1)]
        else:
            state["current_difficulty"] = current_difficulty
        state["hint_count"] = 0
        logging.info(f"Decision: continue (score={last_score}, new_difficulty={state['current_difficulty']})")

    return state

'''def build_graph(agent: InterviewerAgent):
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
    '''