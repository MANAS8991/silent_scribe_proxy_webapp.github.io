import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import speech_recognition as sr
import numpy as np
import wave
import pyaudio
import tempfile
import threading
import time
from io import BytesIO

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize session state
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = ""
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'recorder' not in st.session_state:
    st.session_state.recorder = None
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = None

def generate_interview_response(resume_data, question):
    system_prompt = """You are Gelhi, an advanced AI interview preparation assistant. Your role is to act as the candidate and provide impressive, concise answers that demonstrate:

    1. Deep understanding of the field
    2. Practical experience through STAR method examples
    3. Problem-solving capabilities
    4. Cultural fit and soft skills
    5. Enthusiasm and confidence

    Resume Context:
    {resume_data}

    Generate responses that align with my resume and demonstrate my capabilities as a candidate."""

    try:
        messages = [
            {"role": "system", "content": system_prompt.format(resume_data=resume_data)},
            {"role": "user", "content": f"Interviewer: {question}\n\nGenerate an impressive response that aligns with my resume and demonstrates my capabilities."}
        ]
        
        response = client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.recording = False
        self.rate = 44100
        self.chunk = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_buffer = BytesIO()

    def start_recording(self):
        self.frames = []
        self.recording = True
        
        def record():
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            while self.recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {str(e)}")
                    break
                    
            self.stream.stop_stream()
            self.stream.close()

        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.record_thread.join()

            # Save audio to BytesIO buffer
            with wave.open(self.audio_buffer, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))

            self.audio_buffer.seek(0)  # Reset buffer position for reading
            return self.audio_buffer

    def cleanup(self):
        self.audio.terminate()
        self.audio_buffer.close()

def transcribe_audio(audio_buffer):
    if not audio_buffer:
        return ""
    
    recognizer = sr.Recognizer()
    
    # Adjust recognition settings
    recognizer.energy_threshold = 300  # Increase sensitivity
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8  # Shorter pause detection
    
    try:
        with sr.AudioFile(audio_buffer) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio_data)
            except:
                try:
                    text = recognizer.recognize_sphinx(audio_data)
                except:
                    text = ""
            
            return text.strip()
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""

# Streamlit UI
st.title("Silent Scribe - Interview Preparation & Proxy AI")
st.markdown("""
    📝 Upload your resume and ask interview questions to get expert-level responses Instant!
    Use text or voice input for your questions.
""")

# Resume input section
st.header("My Resume")
resume_input = st.text_area(
    "Paste your resume text here",
    value=st.session_state.resume_data,
    height=200,
    placeholder="Paste your complete resume including experience, skills, and achievements..."
)

if resume_input:
    st.session_state.resume_data = resume_input

# Question input section
st.header("My Question")
input_method = st.radio("Choose input method:", ("Text", "Voice"), key="input_method")

if input_method == "Text":
    st.session_state.question = st.text_input("Type your interview question here", key="text_question")
else:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎤 Start Recording", key="start_recording", disabled=st.session_state.recording):
            if not st.session_state.recorder:
                st.session_state.recorder = AudioRecorder()
            st.session_state.recording = True
            st.session_state.recorder.start_recording()
            st.warning("Recording in progress... Click 'Stop' when finished")
    
    with col2:
        if st.button("⏹️ Stop", key="stop_recording", disabled=not st.session_state.recording):
            if st.session_state.recording and st.session_state.recorder:
                audio_buffer = st.session_state.recorder.stop_recording()
                st.session_state.recording = False
                
                if audio_buffer:
                    with st.spinner("Processing audio..."):
                        text = transcribe_audio(audio_buffer)
                        if text:
                            st.session_state.question = text
                            st.success("Transcribed successfully!")
                            st.write(f"Your question: {text}")
                        else:
                            st.error("No speech detected. Please try again.")
                            
                st.session_state.recorder.cleanup()
    
    with col3:
        if st.button("🔄 Reset", key="reset_recording"):
            if st.session_state.recorder:
                st.session_state.recorder.cleanup()
            st.session_state.recording = False
            st.session_state.question = ""
            st.session_state.recorder = None

# Generate response
response_col1, response_col2 = st.columns([3, 1])
with response_col1:
    if st.button("Generate Response", key="generate_response", 
                disabled=not (st.session_state.question and st.session_state.resume_data)):
        with st.spinner("Generating impressive response..."):
            response = generate_interview_response(st.session_state.resume_data, st.session_state.question)
            if response:
                st.markdown("### Response:")
                st.markdown(response)

with response_col2:
    if st.button("Clear All", key="clear_button"):
        if st.session_state.recorder:
            st.session_state.recorder.cleanup()
        st.session_state.question = ""
        st.session_state.recording = False
        st.session_state.recorder = None
        st.experimental_rerun()

# Sidebar with tips
st.sidebar.title("Interview Tips")
st.sidebar.markdown("""
### Best Practices:
1. Be specific in your resume
2. Include quantifiable achievements
3. Highlight relevant skills
4. Prepare STAR examples

### Question Types:
- Technical Skills
- Behavioral
- Problem Solving
- Culture Fit
- Leadership
""")
