from langchain_core.prompts import PromptTemplate

question_prompt = PromptTemplate(
    input_variables=["topic", "difficulty", "history"],
    template="""You are a professional technical interviewer. Generate a {difficulty} interview question on {topic}. Ensure the question is clear, concise, and tests technical understanding. Provide:
    - The question text.
    - A brief expected answer key (2–3 sentences).
    - A difficulty level (easy, medium, hard).
    Avoid repeating questions from the conversation history: {history}.
    Return the response as a valid JSON object, enclosed in triple backticks:
    ```json
    {{
      "question": "Question text here",
      "answer_key": "Expected answer here",
      "difficulty": "easy/medium/hard"
    }}
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
    Return the response as a valid JSON object, enclosed in triple backticks:
    ```json
    {{
      "score": 0,
      "feedback": "Feedback text here"
    }}
    ```
    Ensure the output is strictly JSON, with no additional text outside the backticks.
    """
)

feedback_prompt = PromptTemplate(
    input_variables=["scores", "feedbacks"],
    template="""You are a technical interviewer providing final feedback after an interview. Based on the scores {scores} and per-question feedback {feedbacks}, summarize the candidate's performance. Highlight:
    - Strengths (e.g., strong accuracy, clear explanations).
    - Areas for improvement (e.g., lack of depth, incorrect concepts).
    Provide a concise summary (3–5 sentences) in a professional tone.
    Return the response as a valid JSON object, enclosed in triple backticks:
    ```json
    {{
      "summary": "Summary text here"
    }}
    ```
    Ensure the output is strictly JSON, with no additional text outside the backticks.
    """
)

hint_prompt = PromptTemplate(
    input_variables=["question", "user_answer", "feedback"],
    template="""You are a technical interviewer assisting a candidate who struggled with the question: "{question}". Their answer was: "{user_answer}", with feedback: "{feedback}". Provide either:
    - A concise hint to guide them toward the correct answer (1–2 sentences).
    - A simpler follow-up question on the same topic to reinforce understanding.
    Decide based on whether the answer shows partial understanding (hint) or significant confusion (follow-up).
    Return the response as a valid JSON object, enclosed in triple backticks:
    ```json
    {{
      "type": "hint" or "follow-up",
      "content": "Hint or follow-up question text here"
    }}
    ```
    Ensure the output is strictly JSON, with no additional text outside the backticks.
    """
)

weight_prompt = PromptTemplate(
    input_variables=["topic", "questions"],
    template="""You are a technical interviewer assigning weights to interview questions on the topic '{topic}'. Based on the following questions, assign a weight to each question reflecting its:
    - Importance: Relevance to core concepts of the topic.
    - Complexity: Difficulty level (easy, medium, hard).
    - Relevance: Practical application in the field.
    The weights should be floats (e.g., 0.30, 0.20) summing to 1.0, with higher weights for more important/complex/relevant questions.
    Questions: {questions}
    Return the response as a valid JSON object, enclosed in triple backticks:
    ```json
    {{
      "weights": [0.30, 0.20, 0.20, 0.20, 0.10]
    }}
    ```
    Ensure the output is strictly JSON, with no additional text outside the backticks, and the weights sum to 1.0.
    """
)