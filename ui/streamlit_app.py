"""
streamlit_app.py — Streamlit frontend for the voice notes summarizer.

Lets the user upload an audio file, sends it to the FastAPI backend,
and displays the transcript and summary side by side.
"""

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/process"

st.set_page_config(page_title="Voice Notes Summarizer", layout="wide")
st.title("Voice Notes Summarizer")
st.caption("Upload an audio file to get a transcript and AI-generated summary.")

uploaded_file = st.file_uploader(
    "Choose an audio file (mp3, m4a, wav)",
    type=["mp3", "m4a", "wav"],
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    initial_prompt = st.text_input(
        "Domain hint (optional)",
        placeholder="e.g. backend, API, frontend, deployment",
        help="Words Whisper commonly mishears in your recordings. Leave blank for general speech.",
    )

    if st.button("Process", type="primary"):
        with st.spinner("Transcribing and summarizing..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"initial_prompt": initial_prompt} if initial_prompt else {},
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()

                    left, right = st.columns(2)

                    with left:
                        st.subheader("Transcript")
                        st.text_area(
                            label="transcript",
                            value=data["transcript"],
                            height=300,
                            label_visibility="collapsed",
                        )

                    with right:
                        st.subheader("Summary")
                        st.write(data["summary"])

                        st.subheader("Action Items")
                        st.write(data["action_items"])

                elif response.status_code == 429:
                    st.warning(response.json().get("detail", "Rate limit reached. Try again shortly."))

                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The audio file may be too long — try a shorter clip.")
