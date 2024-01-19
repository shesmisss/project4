from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, TextAreaField
from wtforms.validators import DataRequired

import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

import google.generativeai as genai

genai.configure(api_key="AIzaSyCPKaACd3xU3HxLTYyOnHpAZyv1XQoLkuI")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": [
        "총 5번의 질문을 할거야.\n\n1 장르 : 성장 로맨스 현대소설\n\n2 작성 규칙 :\n아래의 모든 조건들을 반드시 지켜주세요.\n조건1: 일관성 있는 문장을 만들어줘.\n조건2: 성장로맨스 소설에 어울리는 문체를 유지해줘."]
  },
  {
    "role": "model",
    "parts": [나무를 바라보면 초록색과 주황색이 조화를 이루는 환상적인 그림을 보았습니다."]
  },
])

convo.send_message("YOUR_USER_INPUT")
print(convo.last.text)