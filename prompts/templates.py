from langchain_core.prompts import PromptTemplate

question_prompt = PromptTemplate(
    input_variables=["topic", "difficulty", "history"],
    template="""You are a professional technical interviewer. Generate a {difficulty} interview question on {topic}. Ensure the question is clear, concise, and tests technical understanding. Provide:
    - The question text.
    - A brief expected answer key (2–3 sentences).
    - A difficulty level (easy, medium, hard).
    Avoid repeating questions from the conversation history: {history}.
    Format the output as JSON:
    ```json
    {
      "question": "Question text here",
      "answer_key": "Expected answer here",
      "difficulty": "easy/medium/hard"
    }
    ```
    """
)

evaluation_prompt = PromptTemplate(
    input_variables=["question", "answer_key", "user_answer"],
    template="""You are a technical interviewer evaluating a candidate's response. For the question: "{question}", with expected answer: "{answer_key}", evaluate the user's answer: "{user_answer}". Score the answer (0–10) based on:
    - Accuracy: Correctness of technical content.
    - Clarity: Structure and coherence.
    - Depth: Level of detail and insight.
    Provide:
    - A score (0–10).
    - Brief feedback (2–3 sentences) explaining the score.
    Format the output as JSON:
    ```json
    {
      "score": 0,
      "feedback": "Feedback text here"
    }
    ```
    """
)

feedback_prompt = PromptTemplate(
    input_variables=["scores", "feedbacks"],
    template="""You are a technical interviewer providing final feedback after an interview. Based on the scores {scores} and per-question feedback {feedbacks}, summarize the candidate's performance. Highlight:
    - Average score.
    - Strengths (e.g., strong accuracy, clear explanations).
    - Areas for improvement (e.g., lack of depth, incorrect concepts).
    Provide a concise summary (3–5 sentences) in a professional tone.
    Format the output as JSON:
    ```json
    {
      "summary": "Summary text here",
      "average_score": 0.0
    }
    ```
    """
)