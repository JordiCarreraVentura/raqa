import os
import asyncio

import streamlit as st
from agents import Runner

from raqa._agent import initialize, create_agent


st.set_page_config(page_title="RAQA", page_icon="📚", layout="centered")

data_dir = os.environ.get("RAQA_DATA_DIR", "data")

if "ready" not in st.session_state:
    with st.spinner("Loading documents..."):
        initialize(data_dir=data_dir)
        st.session_state.agent = create_agent()
        st.session_state.messages = []
        st.session_state.history = None
        st.session_state.ready = True

st.title("📚 RAQA")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = st.session_state.history
            input_for = history + [{"role": "user", "content": prompt}] if history else prompt
            result = asyncio.run(Runner.run(st.session_state.agent, input_for))
            st.session_state.history = result.to_input_list()
            response = result.final_output
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
