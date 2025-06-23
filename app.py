import streamlit as st 
from agents.agent import InterviewerAgent
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(filename="interview.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = InterviewerAgent()
if "state" not in st.session_state:
    st.session_state.state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "answer_submitted" not in st.session_state:
    st.session_state.answer_submitted = False
if "hint_requested" not in st.session_state:
    st.session_state.hint_requested = False
if "summary_displayed" not in st.session_state:
    st.session_state.summary_displayed = False

# Streamlit UI
st.title("🤖 AI Interviewer Agent")
st.markdown("Welcome! I'm your AI interviewer. Let's start a technical interview.")

if not st.session_state.state:
    topic = st.text_input("🎯 Enter a technical topic (e.g., Python, Machine Learning):", key="topic_input")
    if st.button("Start Interview 🚀"):
        if topic and len(topic.strip()) >= 3:
            new_state = {
                "topic": topic.strip(),
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
            st.session_state.state = st.session_state.agent.generate_question(new_state)
            st.session_state.chat_history.append({
                "role": "Interviewer",
                "content": f"❓ Question {st.session_state.state['question_count']}: {st.session_state.state['current_question']['question']}"
            })
            st.rerun()
        else:
            st.error("Please enter a valid topic (at least 3 characters).")

# Chat UI
if st.session_state.state and not st.session_state.summary_displayed:
    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role']}**: {msg['content']}")

    if st.session_state.state["decision"] != "end":
        answer_key = f"answer_{st.session_state.state['question_count']}_{st.session_state.state['hint_count']}_{st.session_state.state['is_follow_up']}"
        answer = st.text_area("📝 Your answer:", key=answer_key)

        btn_label = "Submit Interview ✅" if st.session_state.state['question_count'] == 5 else "Next ➡️"
        submit = st.button(btn_label)

        if submit and answer.strip():
            st.session_state.state["current_answer"] = answer.strip()
            st.session_state.chat_history.append({"role": "You", "content": answer.strip()})
            st.session_state.state = st.session_state.agent.evaluate_answer(st.session_state.state)
            score = st.session_state.state["scores"][-1]

            if score >= 7:
                st.session_state.state["decision"] = "continue"
            elif score < 4:
                st.session_state.state = st.session_state.agent.generate_hint(st.session_state.state)
                last_line = st.session_state.state["history"].split("\n")[-2]
                if "Hint:" in last_line:
                    st.session_state.chat_history.append({"role": "Interviewer", "content": f"💡 {last_line}"})
                if st.session_state.state["is_follow_up"]:
                    st.session_state.chat_history.append({
                        "role": "Interviewer",
                        "content": f"🔁 Follow-up Question: {st.session_state.state['current_question']['question']}"
                    })
            else:
                st.session_state.state["decision"] = "continue"

            if st.session_state.state["decision"] == "continue":
                st.session_state.state = st.session_state.agent.generate_question(st.session_state.state)
                st.session_state.chat_history.append({
                    "role": "Interviewer",
                    "content": f"❓ Question {st.session_state.state['question_count']}: {st.session_state.state['current_question']['question']}"
                })

            st.rerun()

    if st.session_state.state["decision"] == "end":
        st.session_state.state = st.session_state.agent.generate_feedback(st.session_state.state)
        st.session_state.summary_displayed = True
        st.rerun()

# Summary display
if st.session_state.summary_displayed:
    st.markdown("### 📊 Interview Summary")
    st.markdown(f"**Final Interview Score: {st.session_state.state['feedback']['final_score']:.1f}/10**")
    for i, (q, a, s, fb, w) in enumerate(zip(
        st.session_state.state["questions"],
        st.session_state.state["answers"],
        st.session_state.state["scores"],
        st.session_state.state["feedbacks"],
        st.session_state.state["feedback"]["weights"]
    )):
        st.markdown(f"#### Question {i+1}")
        st.markdown(f"**❓ Question**: {q['question']}")
        st.markdown(f"**📝 Answer**: {a}")
        st.markdown(f"**✅ Score**: {s}/10")
        st.markdown(f"**⚖️ Weight**: {w:.2f}")
        st.markdown(f"**💬 Feedback**: {fb}")
    st.markdown(f"**🧠 Summary**: {st.session_state.state['feedback']['summary']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"output/interview_{timestamp}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        for msg in st.session_state.chat_history:
            f.write(f"{msg['role']}: {msg['content']}\n\n")
        f.write(f"\nFinal Score: {st.session_state.state['feedback']['final_score']:.1f}/10\n")
        f.write(f"Summary: {st.session_state.state['feedback']['summary']}\n")

    with open(output_path, "r", encoding="utf-8") as f:
        markdown_data = f.read()

    st.download_button("📥 Download Interview Summary", data=markdown_data, file_name=output_path.name)

    st.markdown("---")
    st.success("🎉 Thank you for taking the interview! We hope this feedback helps you grow and prepare better. Good luck! 🚀")
