import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from prompts.templates import question_prompt, evaluation_prompt, feedback_prompt, hint_prompt, weight_prompt
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
        logging.info("Initialized LLM with OpenAI (gpt-4o-mini)")
        self.vector_store = VectorStore()
        self.max_questions = 5
        self.max_hints = 1
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
            "hint_count": 0,
            "follow_up_answers": [],
            "follow_up_scores": [],
            "follow_up_feedbacks": []
        }
        logging.info(f"Selected topic: {topic}")
        logging.debug(f"Initial state: {new_state}")
        self.used_questions.clear()
        return new_state

    def generate_question(self, state: InterviewState) -> InterviewState:
        logging.debug(f"Generating question with state: {state}")
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
        if not state.get("is_follow_up", False):
            state["questions"].append(question)
            state["question_count"] = state.get("question_count", 0) + 1
            state["hint_count"] = 0
        state["history"] = state.get("history", "") + f"Question: {question['question']}\n"
        state["current_question"] = question
        state["is_follow_up"] = False
        state["current_answer"] = ""
        logging.info(f"Generated question (source: {source}, difficulty: {difficulty}): {question['question']}")
        print(f"\nQuestion {state['question_count']}: {question['question']}")
        return state

    def collect_answer(self, state: InterviewState) -> InterviewState:
        logging.debug(f"Collecting answer with state: {state}")
        answer = input("Your answer: ").strip()
        is_follow_up = state.get("is_follow_up", False)
        if is_follow_up:
            state["follow_up_answers"].append(answer)
        else:
            state["answers"].append(answer)
        state["history"] = state.get("history", "") + f"Answer: {answer}\n"
        state["current_answer"] = answer
        logging.info(f"Collected answer for question {state['question_count']}{' (follow-up)' if is_follow_up else ''}: {answer}")
        return state

    def evaluate_answer(self, state: InterviewState) -> InterviewState:
        logging.debug(f"Evaluating answer with state: {state}")
        if not state["current_answer"]:
            evaluation = {
                "score": 0,
                "feedback": "No answer provided for the question."
            }
        else:
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
        is_follow_up = state.get("is_follow_up", False)
        if is_follow_up:
            state["follow_up_scores"].append(evaluation["score"])
            state["follow_up_feedbacks"].append(evaluation["feedback"])
        else:
            state["scores"].append(evaluation["score"])
            state["feedbacks"].append(evaluation["feedback"])
        logging.info(f"Evaluation for question {state['question_count']}{' (follow-up)' if is_follow_up else ''}: Score={evaluation['score']}, Feedback={evaluation['feedback']}")
        return state

    def generate_hint(self, state: InterviewState) -> InterviewState:
        logging.debug(f"Generating hint with state: {state}")
        if state.get("hint_count", 0) >= self.max_hints:
            state["decision"] = "continue"
            state["hint_count"] = 0
            state["is_follow_up"] = False
            state["current_answer"] = ""
            logging.info("Max hints reached for current question, moving to next.")
            return state

        prompt = hint_prompt.format(
            question=state["current_question"]["question"],
            user_answer=state["current_answer"],
            feedback=state["feedbacks"][-1] if state["feedbacks"] else "No feedback available."
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
        state["current_answer"] = ""
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
        
        logging.info(f"Generated {hint_data['type']} for question {state['question_count']}: {hint_data['content']}")
        return state

    def generate_feedback(self, state: InterviewState) -> InterviewState:
        logging.debug(f"Generating feedback with state: {state}")
        # Generate weights
        questions_json = json.dumps([q["question"] for q in state["questions"]])
        prompt = weight_prompt.format(
            topic=state["topic"],
            questions=questions_json
        )
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            logging.info(f"Weights LLM response: {response_text}")
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]
            weights_data = json.loads(response_text)
            weights = weights_data["weights"]
            if len(weights) != len(state["questions"]) or abs(sum(weights) - 1.0) > 0.01:
                weights = [1.0 / len(state["questions"]) for _ in state["questions"]]
        except (json.JSONDecodeError, KeyError):
            weights = [1.0 / len(state["questions"]) for _ in state["questions"]]
        
        # Generate feedback
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
                "summary": "Unable to generate summary due to formatting issue."
            }
        
        # Compute weighted final score
        final_score = sum(s * w for s, w in zip(state["scores"], weights))
        
        print("\n=== Interview Summary ===")
        print(f"Final Interview Score: {final_score:.1f}/10")
        for i, (question, answer, score, feedback_text, weight) in enumerate(zip(
            state["questions"], state["answers"], state["scores"], state["feedbacks"], weights
        )):
            print(f"\nQuestion {i + 1}: {question['question']}")
            print(f"Answer: {answer}")
            print(f"Score: {score}/10")
            print(f"Weight: {weight:.2f}")
            print(f"Feedback: {feedback_text}")
            if i < len(state.get("follow_up_answers", [])):
                print(f"Follow-up Answer: {state['follow_up_answers'][i]}")
                print(f"Follow-up Score: {state['follow_up_scores'][i]}/10")
                print(f"Follow-up Feedback: {state['follow_up_feedbacks'][i]}")
        
        print(f"\nSummary: {feedback['summary']}")
        state["feedback"] = {
            "summary": feedback["summary"],
            "final_score": final_score,
            "weights": weights,
            "follow_up_answers": state.get("follow_up_answers", []),
            "follow_up_scores": state.get("follow_up_scores", []),
            "follow_up_feedbacks": state.get("follow_up_feedbacks", [])
        }
        logging.info(f"Summary: Final Score={final_score:.1f}, Weights={weights}, Summary={feedback['summary']}")
        self.save_interview_output(state)
        return state

    def save_interview_output(self, state: InterviewState):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"interview_{timestamp}.md"
        markdown_content = f"# Interview Summary for {state['topic'].capitalize()} ({timestamp})\n\n"
        markdown_content += f"**Final Interview Score: {state['feedback']['final_score']:.1f}/10**\n\n"
        
        for i, (question, answer, score, feedback, weight) in enumerate(zip(
            state["questions"], state["answers"], state["scores"], state["feedbacks"], state["feedback"]["weights"]
        )):
            markdown_content += (
                f"## Question {i + 1}\n"
                f"**Question**: {question['question']} ({question['difficulty'].capitalize()})\n"
                f"**Answer**: {answer}\n"
                f"**Score**: {score}/10\n"
                f"**Weight**: {weight:.2f}\n"
                f"**Feedback**: {feedback}\n"
            )
            if i < len(state.get("follow_up_answers", [])):
                markdown_content += (
                    f"**Follow-up Answer**: {state['follow_up_answers'][i]}\n"
                    f"**Follow-up Score**: {state['follow_up_scores'][i]}/10\n"
                    f"**Follow-up Feedback**: {state['follow_up_feedbacks'][i]}\n\n"
                )
            else:
                markdown_content += "\n"
        
        markdown_content += (
            f"## Final Summary\n"
            f"**Summary**: {state['feedback']['summary']}\n"
        )
        
        with output_file.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info(f"Saved interview output to {output_file}")

'''import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from prompts.templates import question_prompt, evaluation_prompt, feedback_prompt, hint_prompt, weight_prompt
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
        self.max_hints = 1
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
            "hint_count": 0,
            "follow_up_answers": [],  # Store follow-up answers separately
            "follow_up_scores": [],
            "follow_up_feedbacks": []
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
        if not state.get("is_follow_up", False):
            state["questions"].append(question)
            state["question_count"] = state.get("question_count", 0) + 1
            state["hint_count"] = 0
        state["history"] = state.get("history", "") + f"Question: {question['question']}\n"
        state["current_question"] = question
        state["is_follow_up"] = False
        state["current_answer"] = ""
        logging.info(f"Generated question (source: {source}, difficulty: {difficulty}): {question['question']}")
        print(f"\nQuestion {state['question_count']}: {question['question']}")
        return state

    def collect_answer(self, state: InterviewState) -> InterviewState:
        answer = input("Your answer: ").strip()
        if state.get("is_follow_up", False):
            state["follow_up_answers"].append(answer)
        else:
            state["answers"].append(answer)
        state["history"] = state.get("history", "") + f"Answer: {answer}\n"
        state["current_answer"] = answer
        logging.info(f"Collected answer for question {state['question_count']}{' (follow-up)' if state['is_follow_up'] else ''}: {answer}")
        return state

    def evaluate_answer(self, state: InterviewState) -> InterviewState:
        if not state["current_answer"]:
            evaluation = {
                "score": 0,
                "feedback": "No answer provided for the question."
            }
        else:
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
        if state.get("is_follow_up", False):
            state["follow_up_scores"].append(evaluation["score"])
            state["follow_up_feedbacks"].append(evaluation["feedback"])
        else:
            state["scores"].append(evaluation["score"])
            state["feedbacks"].append(evaluation["feedback"])
        logging.info(f"Evaluation for question {state['question_count']}{' (follow-up)' if state['is_follow_up'] else ''}: Score={evaluation['score']}, Feedback={evaluation['feedback']}")
        return state

    def generate_hint(self, state: InterviewState) -> InterviewState:
        if state.get("hint_count", 0) >= self.max_hints:
            state["decision"] = "continue"
            state["hint_count"] = 0
            state["is_follow_up"] = False
            state["current_answer"] = ""
            logging.info("Max hints reached for current question, moving to next.")
            return state

        prompt = hint_prompt.format(
            question=state["current_question"]["question"],
            user_answer=state["current_answer"],
            feedback=state["feedbacks"][-1] if state["feedbacks"] else "No feedback available."
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
        state["current_answer"] = ""
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
        
        logging.info(f"Generated {hint_data['type']} for question {state['question_count']}: {hint_data['content']}")
        return state

    def generate_feedback(self, state: InterviewState) -> InterviewState:
        # Generate weights
        questions_json = json.dumps([q["question"] for q in state["questions"]])
        prompt = weight_prompt.format(
            topic=state["topic"],
            questions=questions_json
        )
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            logging.info(f"Weights LLM response: {response_text}")
            if response_text.startswith("```json\n") and response_text.endswith("\n```"):
                response_text = response_text[8:-4]
            weights_data = json.loads(response_text)
            weights = weights_data["weights"]
            if len(weights) != len(state["questions"]) or abs(sum(weights) - 1.0) > 0.01:
                weights = [1.0 / len(state["questions"]) for _ in state["questions"]]
        except (json.JSONDecodeError, KeyError):
            weights = [1.0 / len(state["questions"]) for _ in state["questions"]]
        
        # Generate feedback
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
                "summary": "Unable to generate summary due to formatting issue."
            }
        
        # Compute weighted final score
        final_score = sum(s * w for s, w in zip(state["scores"], weights))
        
        print("\n=== Interview Summary ===")
        print(f"Final Interview Score: {final_score:.1f}/10")
        for i, (question, answer, score, feedback_text, weight) in enumerate(zip(
            state["questions"], state["answers"], state["scores"], state["feedbacks"], weights
        )):
            print(f"\nQuestion {i + 1}: {question['question']}")
            print(f"Answer: {answer}")
            print(f"Score: {score}/10")
            print(f"Weight: {weight:.2f}")
            print(f"Feedback: {feedback_text}")
            # Display follow-up if it exists for this question
            if i < len(state["follow_up_answers"]):
                print(f"Follow-up Answer: {state['follow_up_answers'][i]}")
                print(f"Follow-up Score: {state['follow_up_scores'][i]}/10")
                print(f"Follow-up Feedback: {state['follow_up_feedbacks'][i]}")
        
        print(f"\nSummary: {feedback['summary']}")
        state["feedback"] = {
            "summary": feedback["summary"],
            "final_score": final_score,
            "weights": weights,
            "follow_up_answers": state["follow_up_answers"],
            "follow_up_scores": state["follow_up_scores"],
            "follow_up_feedbacks": state["follow_up_feedbacks"]
        }
        logging.info(f"Summary: Final Score={final_score:.1f}, Weights={weights}, Summary={feedback['summary']}")
        self.save_interview_output(state)
        return state

    def save_interview_output(self, state: InterviewState):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"interview_{timestamp}.md"
        markdown_content = f"# Interview Summary for {state['topic'].capitalize()} ({timestamp})\n\n"
        markdown_content += f"**Final Interview Score: {state['feedback']['final_score']:.1f}/10**\n\n"
        
        for i, (question, answer, score, feedback, weight) in enumerate(zip(
            state["questions"], state["answers"], state["scores"], state["feedbacks"], state["feedback"]["weights"]
        )):
            markdown_content += (
                f"## Question {i + 1}\n"
                f"**Question**: {question['question']} ({question['difficulty'].capitalize()})\n"
                f"**Answer**: {answer}\n"
                f"**Score**: {score}/10\n"
                f"**Weight**: {weight:.2f}\n"
                f"**Feedback**: {feedback}\n"
            )
            if i < len(state["follow_up_answers"]):
                markdown_content += (
                    f"**Follow-up Answer**: {state['follow_up_answers'][i]}\n"
                    f"**Follow-up Score**: {state['follow_up_scores'][i]}/10\n"
                    f"**Follow-up Feedback**: {state['follow_up_feedbacks'][i]}\n\n"
                )
            else:
                markdown_content += "\n"
        
        markdown_content += (
            f"## Final Summary\n"
            f"**Summary**: {state['feedback']['summary']}\n"
        )
        
        with output_file.open("w", encoding="utf-8") as f:
            f.write(markdown_content)
        logging.info(f"Saved interview output to {output_file}")
'''

