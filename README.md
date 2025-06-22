# 🤖 AI Interviewer Agent

Welcome to the **AI Interviewer Agent** – a langgraph orchestrated agentic system that simulates a technical interview experience.  
Built by **Gunavathi** an AI Developer

---

## 🚀 Features

- **Topic Selection**: Choose a topic (e.g., Machine Learning, Python).
- **Question Generation**: Pulled from ChromaDB or generated via OpenAI's `gpt-4o-mini`.
- **Adaptive Difficulty**: Dynamically adjusts (easy → medium → hard) based on performance.
- **Hints & Follow-Ups**: Get up to 1 hint or follow-up per main question.
- **Weighted Scoring**: Scores are weighted based on relevance using LLM-evaluated weights.
- **Markdown Output**: Final summary saved as `output/interview_<timestamp>.md`.

---

## ⚙️ Setup Instructions

### ✅ Prerequisites
- Python 3.8+
- OpenAI API Key
- (Recommended) Virtual Environment

### 📦 Installation


# Clone the repository
git clone https://github.com/Gunavathi/AI-Interviewer-Agent.git
cd AI-Interviewer-Agent

# Create and activate virtual environment
python -m venv venv
# Windows
venv\\Scripts\\activate
# Linux/Mac
source venv/bin/activate

# Install required packages
pip install -r requirements.txt


### 📄 Environment Variables
Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key_here


### 📁 Verify Data

Ensure data/questions.json contains example questions (e.g., Python, ML). The LLM will auto-generate additional questions if needed.


## 🧠 Technologies Used

- Python: Core language

- LangChain: For LLM integration and prompt handling

- LangGraph: For Orchestration

- ChromaDB: Vector store for question retrieval

- OpenAI GPT-4o-mini: Question generation and evaluation

- Pydantic: Type-safe state management (InterviewState) and for data validation

- Pathlib: File handling

- Dotenv: API key management


## 🧩 Design Decisions & Flow

#### ✅ Highlights

- Modular Architecture: Clean separation of agents, prompts, utils, and workflow

- Hybrid Question Source: First try vector DB; fallback to LLM

- Resilient Parsing: Handles JSON decode errors gracefully

- Markdown Output: For portability and clarity

- State Tracking: Full interview history stored in InterviewState

## 🔁 Workflow via LangGraph (workflow/graph.py)
final_workflow.png


### 📊 Optional Features Implemented
- ✅ Weighted Scoring (sum = 1.0)

- ✅ Adaptive Difficulty (question level adjusts per response)

- ✅ Follow-up Tracking (stored separately from main answers)

- ✅ Markdown Report Generation

- ✅ Score Chart (bar chart of per-question scores in Streamlit)

- ✅ Logging (stored in interview.log)

## 🧪 Usage

### ▶️ Run the App

#### Activate venv
venv\Scripts\activate

#### Start Interview (CLI mode)
python main.py

#### 💬 Interact
Enter a topic (e.g., data structures)

Answer questions (with optional hints)

After 5 questions, view your final report and scores

#### 📁 Output
Markdown summary saved in: output/interview_<timestamp>.md

Debug logs in: interview.log