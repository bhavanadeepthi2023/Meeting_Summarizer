import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Meeting Summarization Engine", layout="wide")
st.title("Meeting Summarization Engine")

uploaded_file = st.file_uploader(
    "Upload Audio Recording",
    type=["mp3", "wav", "m4a", "aac", "ogg", "flac"],
)

if uploaded_file and st.button("Process Audio"):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }

    try:
        response = requests.post(f"{API_URL}/api/v1/jobs", files=files, timeout=60)
        response.raise_for_status()
        job_id = response.json()["job_id"]
    except requests.RequestException as err:
        st.error(f"Failed to submit processing job: {err}")
        st.stop()

    status_placeholder = st.empty()
    progress_bar = st.progress(10)

    while True:
        try:
            poll_resp = requests.get(f"{API_URL}/api/v1/jobs/{job_id}", timeout=10)
            poll_resp.raise_for_status()
            data = poll_resp.json()
        except requests.RequestException as err:
            st.error(f"Polling failure: {err}")
            break

        status = data.get("status")

        if status == "processing_audio":
            status_placeholder.text("Status: Normalizing and chunking audio...")
            progress_bar.progress(30)
        elif status == "transcribing":
            status_placeholder.text("Status: Running transcription engine...")
            progress_bar.progress(60)
        elif status == "extracting_insights":
            status_placeholder.text("Status: Extracting structured tasks and summaries...")
            progress_bar.progress(85)
        elif status == "completed":
            status_placeholder.empty()
            progress_bar.empty()

            result = data["result"]
            st.header(result["meeting_title"])

            st.subheader("Executive Summary")
            st.write(result["executive_summary"])

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Key Decisions")
                for item in result["decisions"]:
                    st.markdown(f"- **{item['decision']}**")
                    if item.get("rationale"):
                        st.caption(f"Rationale: {item['rationale']}")

            with col_right:
                st.subheader("Key Topics")
                for topic in result["key_topics"]:
                    st.markdown(f"- {topic}")

            st.subheader("Action Items")
            st.dataframe(
                result["action_items"],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Raw Transcript"):
                st.text_area(
                    label="Transcript Content",
                    value=data.get("transcript", ""),
                    height=250,
                    disabled=True,
                )
            break
        elif status == "failed":
            status_placeholder.empty()
            progress_bar.empty()
            st.error(f"Processing failed: {data.get('error')}")
            break

        time.sleep(2)