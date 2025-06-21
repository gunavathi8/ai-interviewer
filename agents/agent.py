import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from prompts.templates import question_prompt, evaluation_prompt, feedback_prompt, hint_prompt
from utils.vector_store import VectorStore
from utils.types import InterviewState

logging.basicConfig(
    filename="interview.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

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
        self.max_hints = 1  # Limit to 1 hint/follow-up per question
        self.used_questions = set()
        self.difficulty_levels = ["easy", "medium", "hard"]

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
            "decision": "",
            "current_difficulty": "easy",
            "is_follow_up": False,
            "hint_count": 0
        }
        logging.info(f"Selected topic: {topic}")
        self.used_questions.clear()
        return new_state

    def generate_question(self, state: InterviewState) -> InterviewState:
        if state.get("question_count", 0) >= self.max_questions and not state.get("is_follow_up", False):
            state["decision"] = "end"
            logging.info("Max questions reached.")
            return state

        difficulty = state.get("current_difficulty", "easy")
        if state.get("is_follow_up", False):
            difficulty = max("easy", self.difficulty_levels[max(0, self.difficulty_levels.index(difficulty) - 1)])

        question = self.vector_store.retrieve_question(state["topic"], difficulty)
        source = "vector_store"
        if question and question["question"] in self.used_questions:
            question = None
        if not question:
            source = "llm"
            prompt = question_prompt.format(
                topic=state["topic"],
                difficulty=difficulty,
                history=state.get("history", "")
            )
            try:
                response = self.llm.invoke(prompt)
                question_text = response.content.strip()
                if question_text.startswith("```json\n") and question_text.endswith("\n```"):
                    question_text = question_text[8:-4]
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
        if not state.get("is_follow_up", False):
            state["question_count"] = state.get("question_count", 0) + 1
            state["hint_count"] = 0  # Reset hint count for new question
        state["history"] = state.get("history", "") + f"Question: {question['question']}\n"
        state["current_question"] = question
        state["is_follow_up"] = False
        logging.info(f"Generated question (source: {source}, difficulty: {difficulty}): {question['question']}")
        print(f"\nQuestion {state['question_count']}: {question['question']}")
        return state

    def collect_answer(self, state: InterviewState) -> InterviewState:
        answer = input("Your answer: ").strip()
        state["answers"].append(answer)
        state["history"] = state.get("history", "") + f"Answer: {answer}\n"
        state["current_answer"] = answer
        print(f"Answer: {answer}")
        logging.info(f"Collected answer: {answer}")
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
            logging.info(f"Evaluation LLM response: {response_text}")
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]
            evaluation = json.loads(response_text)
        except json.JSONDecodeError:
            evaluation = {
                "score": 0,
                "feedback": "Unable to evaluate answer due to formatting issue. Please ensure your answer addresses the question clearly."
            }
        state["scores"].append(evaluation["score"])
        state["feedbacks"].append(evaluation["feedback"])
        logging.info(f"Evaluation: Score={evaluation['score']}, Feedback={evaluation['feedback']}")
        return state

    def generate_hint(self, state: InterviewState) -> InterviewState:
        if state.get("hint_count", 0) >= self.max_hints:
            state["decision"] = "continue"
            state["hint_count"] = 0
            state["is_follow_up"] = False
            logging.info("Max hints reached for current question, moving to next.")
            return state

        prompt = hint_prompt.format(
            question=state["current_question"]["question"],
            user_answer=state["current_answer"],
            feedback=state["feedbacks"][-1]
        )
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            logging.info(f"Hint LLM response: {response_text}")
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]
            hint_data = json.loads(response_text)
        except json.JSONDecodeError:
            hint_data = {
                "type": "hint",
                "content": "Please consider the key concepts related to the question and provide more detail."
            }
        
        state["hint_count"] = state.get("hint_count", 0) + 1
        if hint_data["type"] == "hint":
            print(f"\nHint: {hint_data['content']}")
            state["history"] = state.get("history", "") + f"Hint: {hint_data['content']}\n"
        else:
            state["current_question"] = {
                "question": hint_data["content"],
                "answer_key": "Provide a clear answer to the follow-up question.",
                "difficulty": "easy"
            }
            state["is_follow_up"] = True
            state["history"] = state.get("history", "") + f"Follow-up Question: {hint_data['content']}\n"
            print(f"\nFollow-up Question: {hint_data['content']}")
        
        logging.info(f"Generated {hint_data['type']}: {hint_data['content']}")
        return state

    def generate_feedback(self, state: InterviewState) -> InterviewState:
        prompt = feedback_prompt.format(
            scores=state["scores"],
            feedbacks=state["feedbacks"]
        )
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            logging.info(f"Feedback LLM response: {response_text}")
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]
            feedback = json.loads(response_text)
        except json.JSONDecodeError:
            feedback = {
                "summary": "Unable to generate summary due to formatting issue.",
                "average_score": sum(state["scores"]) / len(state["scores"]) if state["scores"] else 0.0
            }
        
        print("\n=== Interview Summary ===")
        for i, (question, answer, score, feedback_text) in enumerate(zip(
            state["questions"], state["answers"], state["scores"], state["feedbacks"]
        )):
            print(f"Question {i + 1}: {question['question']}")
            print(f"Answer: {answer}")
            print(f"Score: {score}/10")
            print(f"Feedback: {feedback_text}\n")
        
        print(f"Average Score: {feedback['average_score']:.2f}")
        print(f"Summary: {feedback['summary']}")
        state["feedback"] = feedback
        logging.info(f"Summary: Average Score={feedback['average_score']}, Summary={feedback['summary']}")
        self.save_interview_output(state)
        return state

    def save_interview_output(self, state: InterviewState):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"interview_{timestamp}.md")
        markdown_content = f"# Interview Summary for {state['topic'].capitalize()} ({timestamp})\n\n"
        
        for i, (question, answer, score, feedback) in enumerate(zip(
            state["questions"], state["answers"], state["scores"], state["feedbacks"]
        )):
            markdown_content += (
                f"## Question {i + 1}\n"
                f"**Question**: {question['question']} ({question['difficulty'].capitalize()})\n"
                f"**Answer**: {answer}\n"
                f"**Score**: {score}/10\n"
                f"**Feedback**: {feedback}\n\n"
            )
        
        markdown_content += (
            f"## Final Summary\n"
            f"**Average Score**: {state['feedback']['average_score']:.2f}\n"
            f"**Summary**: {state['feedback']['summary']}\n"
        )
        
        with output_file.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info(f"Saved interview output to {output_file}")

