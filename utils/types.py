from typing import Any, Dict, List, TypedDict

class InterviewState(TypedDict):
    topic: str
    question_count: int
    questions: List[Dict[str, Any]]
    answers: List[str]
    scores: List[int]
    feedbacks: List[str]
    history: str
    current_question: Dict[str, Any]
    current_answer: str
    decision: str
    

'''class InterviewState(TypedDict):
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
    '''