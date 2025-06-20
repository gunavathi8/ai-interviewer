from typing import TypedDict, List

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