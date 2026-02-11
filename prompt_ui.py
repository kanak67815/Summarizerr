from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
from langchain_core.prompts import PromptTemplate,load_prompt
import os
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("Hugging Face Token not found! Check your .env file.")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token, # Explicitly pass it here
)
model=ChatHuggingFace(llm=llm)
st.header("RESEARCH TOOL")
paper_input=st.selectbox("Select Research paper name",["Attention is all you need","BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding","GPT-3: Language Models are Few-Shot Learners" ])
style_input=st.selectbox("Select the Explanation Style",["Beginner-Friendly","Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
template=load_prompt('template.json')

if st.button("Summarize"):
    prompt = template.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })

    result = model.invoke(prompt)
    st.write(result.content)