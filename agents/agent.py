import json
from langchain_openai import ChatOpenAI
from prompts.templates import question_prompt, evaluation_prompt, feedback_prompt
from utils.vector_store import VectorStore
import os
from dotenv import load_dotenv

load_dotenv()

class InterviewerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
        )
        self.vector_store = VectorStore()
        self.max_questions = 5

    def select_topic(self, state):
        topic = input("Enter a technical topic for the interview (e.g., Python, Data Structures): ").strip()
        while not topic or len(topic) < 3:
            print("Invalid topic. Please enter a valid topic (at least 3 characters).")
            topic = input("Enter a technical topic for the interview: ").strip()
            return {
                "topic": topic,
                "question_count": 0,
                "questions": [],
                "answers": [],
                "scores": [],
                "feedbacks": [],
                "history": ""
            }
        
    def generate_question(self, state):
        difficulty = "easy" if state["question_count"] < 2 else "medium" if state["question_count"] < 4 else "hard"
        question = self.vector_store.retrieve_question(state["topic"], difficulty)
        if not question:
            prompt = question_prompt.format(
                topic=state["topic"],
                difficulty=difficulty,
                history=state["history"]
            )
            response = self.llm.invoke(prompt)
            question = json.loads(response.content)
        state["questions"].append(question)
        state["question_count"] += 1
        state["history"] += f"Question: {question['question']}\n"
        return {"current_question": question}
    
    def collect_answer(self, state):
        print(f"Question {state['question_count']}: {state['current_question']['question']}")
        answer = input("Your answer: ").strip()
        state["answers"].append(answer)
        state["history"] += f"Answer: {answer}\n"
        return {"current_answer": answer}
    
    def evaluate_answer(self, state):
        prompt = evaluation_prompt.format(
            question=state["current_question"]["question"],
            answer_key=state["current_question"]["answer_key"],
            user_answer=state["current_answer"]
        )
        response = self.llm.invoke(prompt)
        evaluation = json.loads(response.content)
        state["scores"].append(evaluation["score"])
        state["feedbacks"].append(evaluation["feedback"])
        print(f"\nFeedback: {evaluation['feedback']}")
        print(f"Score: {evaluation['score']}/10")
        return {"scores": state["scores"], "feedbacks": state["feedbacks"]}
    
    def generate_feedback(self, state):
        prompt = feedback_prompt.format(
            scores=state["scores"],
            feedbacks=state["feedbacks"]
        )
        response = self.llm.invoke(prompt)
        feedback = json.loads(response.content)
        print("\n=== Interview Summary ===")
        print(f"Average Score: {feedback['average_score']:.2f}")
        print(f"Summary: {feedback['summary']}")
        return {"feedback": feedback}

