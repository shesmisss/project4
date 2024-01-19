from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, TextAreaField
from wtforms.validators import DataRequired
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
import markdown


app = Flask(__name__)

load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
# csrf = CSRFProtect(app)

genai.configure(api_key=API_KEY)
generation_config = {
  "temperature": 0.0,
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
        "총 5번의 질문을 할거야.\n\n1 장르 : 성장 로맨스 현대소설\n\n2 작성 규칙 :\n아래의 모든 조건들을 반드시 지켜주세요.\n조건1: 일관성 있는 문장을 만들어줘.\n조건2: 성장로맨스 소설에 어울리는 문체를 유지해줘.\n조건3: 세상을 바라보는 다양한 시각을 보여줘.\n조건4: 첫사랑의 감정을 섬세하게 그려줘.\n조건5: 두 주인공의 성장을 통해, 사랑의 힘을 느끼게 해줘.\n조건6: 따옴표나 쌍따옴표가 나오기 전과 후 한 줄 띄어쓰기 해줘.\n조건7: 전체적으로 따뜻하고 포근한 느낌이야.\n조건8: 영어, 중국어, 특수문자를 제외해줘. 처음부터 끝까지 한글로 써줘.\n\n5. 구성\n발단 : 매일매일 세상을 다른 색깔로 보는 예린이에 대해 설명한다.\n전개 : 같은 반 친구 호수가 예린이와 예린이의 그림에 관심을 가진다. 그리고 세상을 다른 색깔로 보는 예린이가 스스로 긍정적으로 생각할 수 있는 좋은 영향을 끼친다.\n위기 : 예린이와 호수는 서로에게 풋풋한 첫사랑을 느낀다. \n절정 : 같이 바다에 가서 좋은 시간을 보내지만, 안타깝게도 호수의 유학 소식을 듣는다.\n결말 : 둘 다 성인이 되어 예린이의 작품 전시회때 호수가 장미꽃다발을 가져와 고백한다.\n\n\n중 발단 부분을 써줘.  토큰수 2048개를 사용해서.\n소설 생성을 하기 전에 생각의 사슬을 사용해줘."]
  },
  {
    "role": "model",
    "parts": ["**생각의 사슬:**\n\n**[예린]** -> **[세상을 다채롭게 보는 소녀]** -> **[호수]** -> **[예린이의 독특함에 관심을 가짐]** -> **[예린이의 긍정적 영향]** -> **[호수의 변화]** -> **[풋풋한 첫사랑]**\n\n**[발단]**\n\n어느 날, 세상을 다른 색깔로 보는 예린이가 있었습니다. 그녀에게 세상은 무지개처럼 다채로웠고, 매일매일이 새로운 색으로 물들어 있었습니다. 그녀는 하늘을 쳐다보면 푸른색과 노란색이 어우러진 경이로운 풍경을 보았고, 나무를 바라보면 초록색과 주황색이 조화를 이루는 환상적인 그림을 보았습니다. 심지어 사람들을 바라보면 각자의 얼굴에서 고유한 색깔을 느꼈습니다. 예린이는 마치 세상을 거대한 팔레트로 여기고, 매일매일 다른 색깔을 섞어 새로운 그림을 그리고 있는 것처럼 느꼈습니다.\n\n그녀는 이런 자신의 독특한 시각으로 인해 다른 아이들과는 약간 달랐습니다. 친구들은 그녀가 보는 세상이 어떤지 이해하지 못했고, 그녀를 이상하게 여기기도 했습니다. 그러나 예린이는 자신의 눈으로 보는 세상이 너무나도 아름다웠고, 그것을 바꿀 생각은 전혀 없었습니다. 그녀는 세상을 다채롭게 보는 것이 자신의 특별함이라고 생각했고, 그것을 자랑스럽게 여겼습니다.\n\n어느 날, 같은 반 친구 호수가 예린이의 특별함을 알아차렸습니다. 그는 예린이가 세상을 보는 방식이 다른 아이들과는 확연히 다르다는 것을 알아차렸고, 그녀의 그림에 깊은 인상을 받았습니다. 예린이는 자신의 그림을 통해 세상의 다채로움을 표현했고, 호수는 그 그림에서 세상을 보는 새로운 시각을 얻었습니다. 그는 예린이가 자신의 세상을 바라보는 방식을 바꾸어 놓은 특별한 소녀라는 것을 깨달았고, 그녀에게 끌리기 시작했습니다."]
  },
])



class NovelForm(FlaskForm):
    characters = StringField('등장인물', validators=[DataRequired()])
    setting = TextAreaField('구체적 설정')
    genre = SelectField('장르를 선택하세요', choices=[('romance', '로맨스'), ('thriller', '스릴러'), ('fantasy', '판타지')])
    dialect = SelectField('사투리선택', choices=[('standard', '표준어'), ('busan', '부산사투리')])
    submit = SubmitField('생성하기')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/page1', methods=['GET', 'POST'])
def page1():
    form = NovelForm()
    novel = ""  # 'novel' 변수 초기화
    if form.validate_on_submit():
        characters = f"등장인물: {form.characters.data}"
        setting = f"설정: {form.setting.data}"
        genre = f"장르: {form.genre.data}"
        dialect = f"사투리: {form.dialect.data}"

        # 모델에 전달할 입력 준비
        user_input = f"{characters}, {setting}, {genre}, {dialect}"

        # 모델 대화 시작
        model.start_chat(history=[{"role": "user", "parts": [user_input]}])

        # 모델에 사용자 입력 전달 및 응답 받기
        convo.send_message(user_input)
        novel = convo.last.text


        html_content = markdown.markdown(novel)
        return render_template('page1.html', form=form, novel=html_content)  # form 객체 전달

    return render_template('page1.html', form=form, novel="")  # form 객체 전달


# def generate_novel(characters, setting, genre, dialect):
#     # LangChain과 LLM을 사용하여 소설을 생성하는 로직 구현
#     return f"생성된소설은 {characters}, {setting}, {genre}, {dialect}"




@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')




if __name__ == '__main__':
    app.run(debug=True)
