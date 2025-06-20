import json
import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from prompts.templates import question_prompt, evaluation_prompt, feedback_prompt
from utils.vector_store import VectorStore
from utils.types import InterviewState

load_dotenv()

class InterviewerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )
        self.vector_store = VectorStore()
        self.max_questions = 5
        self.used_questions = set()

    def select_topic(self, state: InterviewState) -> InterviewState:
        topic = input("Enter a technical topic for the interview (e.g., Python, Data Structures): ").strip()
        while not topic or len(topic) < 3:
            print("Please enter a valid topic (at least 3 characters).")
            topic = input("Enter a technical topic: ").strip()
        new_state = {
            "topic": topic,
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
        print(f"Selected topic: {topic}")
        self.used_questions.clear()  # Reset used questions for new session
        return new_state

    def generate_question(self, state: InterviewState) -> InterviewState:
        print(f"Generating question. Current state: {state}")
        if state.get("question_count", 0) >= self.max_questions:
            print("Max questions reached.")
            state["decision"] = "end"
            return state
        
        difficulty = (
            "easy" if state.get("question_count", 0) < 2
            else "medium" if state.get("question_count", 0) < 4
            else "hard"
        )
        question = self.vector_store.retrieve_question(state["topic"], difficulty)
        if question and question["question"] in self.used_questions:
            question = None  # Skip repeated question
        if not question:
            prompt = question_prompt.format(
                topic=state["topic"],
                difficulty=difficulty,
                history=state.get("history", "")
            )
            try:
                response = self.llm.invoke(prompt)
                question_text = response.content.strip()
                if question_text.startswith("```json\n") and question_text.endswith("\n```"):
                    question_text = question_text[8:-4]  # Remove ```json\n and \n```
                question = json.loads(question_text)
                if question["question"] in self.used_questions:
                    question = {
                        "question": f"Explain a {difficulty} concept in {state['topic']}.",
                        "answer_key": f"Provide a detailed explanation of a {difficulty} concept in {state['topic']}.",
                        "difficulty": difficulty
                    }
            except json.JSONDecodeError:
                question = {
                    "question": f"Explain a {difficulty} concept in {state['topic']}.",
                    "answer_key": f"Provide a detailed explanation of a {difficulty} concept in {state['topic']}.",
                    "difficulty": difficulty
                }
        
        self.used_questions.add(question["question"])
        state["questions"].append(question)
        state["question_count"] = state.get("question_count", 0) + 1
        state["history"] = state.get("history", "") + f"Question: {question['question']}\n"
        state["current_question"] = question
        print(f"Generated question: {question['question']}")
        return state

    def collect_answer(self, state: InterviewState) -> InterviewState:
        print(f"\nQuestion {state['question_count']}: {state['current_question']['question']}")
        answer = input("Your answer: ").strip()
        state["answers"].append(answer)
        state["history"] = state.get("history", "") + f"Answer: {answer}\n"
        state["current_answer"] = answer
        print(f"Collected answer: {answer}")
        return state

    def evaluate_answer(self, state: InterviewState) -> InterviewState:
        prompt = evaluation_prompt.format(
            question=state["current_question"]["question"],
            answer_key=state["current_question"]["answer_key"],
            user_answer=state["current_answer"]
        )
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            print(f"Raw LLM response: {response_text}")  # Debug LLM output
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]  # Remove ```json\n and \n```
            evaluation = json.loads(response_text)
        except json.JSONDecodeError:
            evaluation = {
                "score": 0,
                "feedback": "Unable to evaluate answer due to formatting issue. Please ensure your answer addresses the question clearly."
            }
        state["scores"].append(evaluation["score"])
        state["feedbacks"].append(evaluation["feedback"])
        print(f"\nFeedback: {evaluation['feedback']}")
        print(f"Score: {evaluation['score']}/10")
        return state

    def generate_feedback(self, state: InterviewState) -> InterviewState:
        prompt = feedback_prompt.format(
            scores=state["scores"],
            feedbacks=state["feedbacks"]
        )
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            print(f"Raw LLM feedback response: {response_text}")  # Debug LLM output
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]  # Remove ```json\n and \n```
            feedback = json.loads(response_text)
        except json.JSONDecodeError:
            feedback = {
                "summary": "Unable to generate summary due to formatting issue.",
                "average_score": sum(state["scores"]) / len(state["scores"]) if state["scores"] else 0.0
            }
        print("\n=== Interview Summary ===")
        print(f"Average Score: {feedback['average_score']:.2f}")
        print(f"Summary: {feedback['summary']}")
        state["feedback"] = feedback
        return state