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

## J 코드 시작
''' 
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
# csrf = CSRFProtect(app)

# 제미나이 컨피그 설정
genai.configure(api_key=API_KEY)
generation_config = {
  "temperature": 0.0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

# 제미나이 세이프티 세팅
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

# 모델불러오기
model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

'''
## J 코드 끝


## 1Q님 코드 삽입 시작

# Python : 3.10.13
# Created: JAN. 08. 2024
# Author: 1.Q. CHOI
# 소설쓰기 전문가 버전3 (writeNovel_Expert_v3)


# 로컬에 아래의 패키지를 모두 설치하였기 때문에 주석처리
# !pip install -q --upgrade google-generativeai

#로컬버전으로 변경하였기 때문에 주석처리
#from google.colab import userdata

import getpass
import os
import google.generativeai as genai

# 실행하기 전에 열쇠창에 API키를 등록하세요.
'''
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

model = genai.GenerativeModel('gemini-pro')
'''

# 코랩버전이 아니므로 주석처리 로컬 환경변수에서 불러오는 것으로 변경

# Gemini 설정

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name = "gemini-pro")


import time
import re
from tqdm.notebook import tqdm
from google.generativeai import types as genai

# 소설 장르를 입력하세요.
prompt_genre = """{
소설 장르는 현대 소설이다.
}"""

# 작성 규칙을 입력하세요.
prompt_rule = """{
아래의 모든 조건들을 반드시 지키시오.
조건1 : 반드시 스토리가 이어질 수 있는 문장으로 마무리 하시오.
조건2 : 마지막에 구두점(.)이나 구두점+따옴표로 문장을 마무리 하시오.
조건3 : 'HARM_CATEGORY'에 속하는 문장을 절대로 생성하지 마시오.
조건4 : 스토리의 결말을 지어서는 안됨. 결말을 암시해서도 안됨.
조건5 : 줄바꿈을 적절히 활용하시오. 줄바꿈은 연속되면 안되고, 한번씩만 적용하시오.
조건6 : 대명사 사용을 자제하시오.
조건7 : 언어는 한국어 이외에 절대로 사용 금지.
조건8 : 비슷한 문장 반복 금지. 동일 단어 반복도 금지.
조건9 : 특수문자 절대로 사용금지. 동일 특수문자 연속사용도 금지.
조건10 : 길이가 너무 짧은 문장을 반복하면 안됨.
조건11 : 소재나 주제가 무의미하게 반복될경우, 인접한 다른 소재나 주제로 넘어갈 것. 그것도 한계에 다다르면 새로운 소재나 주제를 등장시킬 것.
조건12 : '행복', '해피', '용기', '사랑받았다', '사랑을 받았다', '살아갈', '프롬프트', '영웅' 같은 단어 사용금지.
조건13 : 영어, 중국어, 일본어, 숫자 절대 사용금지.
조건14 : 등장인물을 최대한 많이 등장시키고 대화를 최대한 많이 삽입하시오.
}"""

# 등장인물을 입력하세요.
prompt_char = """{
절대 사망하지 않는 주인공들은 다음과 같다:
이현우 (32세, 회사원): 현대 도시에서 사는 평범한 직장인. 일상의 소소한 이야기를 중심으로 사랑과 가족, 직장에서의 이야기를 다룸.
박지은 (28세, 디자이너): 성공한 디자이너로서의 삶과 그녀의 성장 이야기를 다룸. 인간관계와 예술적인 도전을 중심으로 풀어낸다.
김태영 (35세, 변호사): 현대 도시의 법정에서 활동하는 변호사로, 복잡한 사건과 고객들과의 상호작용을 다루며 성장하는 이야기를 그린다.
조현서 (25세, 대학생): 청춘의 시작, 대학생활과 그녀의 친구들과의 이야기를 중심으로 새로운 경험과 성장을 그린다.
강민호 (40세, 음식점 사업가): 현대 도시의 맛집을 운영하는 사업가로, 음식과 사업, 가족과의 교감을 다루는 이야기를 전개한다.
이정우 (30세, 프리랜서 작가): 글쓰기에 전념하는 프리랜서 작가로, 창작의 고통과 성공, 그리고 인간관계를 그리며 성장하는 이야기를 풀어낸다.
그 밖에도 다양한 등장인물을 등장시키시오.
}"""

# 시간-공간적 배경을 입력하세요.
prompt_back = """{
이야기의 배경은 다양한 현대 도시와 지역에서 전개된다. 번화한 상업지구부터 한적한 주택가, 현대 예술 갤러리나 공원까지 다양한 장면들이 소설에 등장한다.
일상 속에서 벌어지는 다양한 사건들과 상황들이 독자를 현대 도시의 다양성과 색다른 경험으로 안내한다.
}"""

# 시작을 위한 지시사항을 입력하세요.
prompt_start = """{
당신은 한국의 저명한 소설가이다. 당신에게 소설을 작성해달라는 거액의 부탁이 들어와서 소설 작성 작업을 한다.
당신의 모든 에너지와 집중력을 발휘해서 다음에 제시되는 프롬프트의 내용에 이어지는 내용을 최대한 매끄러운 내용으로 작성한다.
절대로 어색한 문장이나 똑같이 반복되는 내용을 추가해서는 안된다. 제시받은 프롬프트의 내용이 반복되어서는 절대 안된다.
또한 특수문자를 사용하지 마시오. 그리고 소설내용을 추가할 때 심사숙고하여 앞의 내용과 중복되는 내용을 절대로 작성하지 마시오.
반드시 인물들의 대화를 최대한 많이 삽입하시오. 대화내용이 스토리 전개의 중심입니다. 전체적인 스토리전개는 아주아주 조금씩 디테일하고 긴 호흡으로 전개되도록 하시오.
장면의 묘사를 최대한 디테일하고 실감나게 표현 및 서술하시오. 프롬프트에 대한 언급 금지.
}"""

# 시작 지시사항을 입력하세요.
prompt_action = """{
소설을 맨 처음부터 작성할 것이다. 첫 시작을 매끄럽게 만들 것.
}"""

# 마무리를 위한 지시사항을 입력하세요.
prompt_end = """{
이어지는 내용으로 반드시 소설작성을 마무리 하시오. 또한 아래의 조건을 반드시 준수하시오.
조건1 : 진행되고 있던 소설을 급하지 않게, 그리고 아주 매끄럽게 마무리 하시오. 확실하게 결말을 지으시오. 마무리 이후에는 더 이상의 아무런 문장도 생성하지 않을 것.
조건2 : 마지막에 구두점(.)으로 문장을 마무리 하시오.
조건3 : 'HARM_CATEGORY'에 속하는 문장을 절대로 생성하지 마시오.
조건4 : 줄바꿈을 적절히 활용하시오. 줄바꿈은 연속되면 안되고, 한번씩만 적용하시오.
조건5 : 언어는 한국어 이외에 절대로 사용 금지.
조건6 : 비슷한 문장 반복 최대한 자제할것. 동일 단어 반복 최대한 자제할것.
조건7 : 인물들간에 대화를 적절히 삽입 하시오.
조건8 : 특수문자 절대로 사용금지. 동일 특수문자 연속사용도 금지.
조건9 : 길이가 너무 짧은 문장을 반복하면 안됨.
조건10 : 영어, 중국어, 숫자 절대 사용금지.
}"""


# 실행횟수를 입력하세요.
n = 3
prompt = ""
myNovel = ""
current_prompt = ""
total_tokens_accumulated = 0
prompt_sum = prompt_genre + '\n' + prompt_char + '\n' + prompt_back + '\n' + prompt_rule + '\n' + prompt_start + '\n'
generation_config = genai.GenerationConfig(temperature=1.0)
keyword_exclude = ['"', '또', '하하', '하지만', '안녕', '말했다']
keyword_banned = ['행복', '해피', '용기', '사랑받았다', '사랑을 받았다', '살아갈', '프롬프트', '영웅']
noise_except = [
    '년', '월', '일', '대', '살',
    '시간', '분', '초', '원',
    '개', '번', '장', '회', '천',
    '만', '백', '척', '마리', '그릇'
]

def replace_strings(input_text):
    replaced_text = (
        input_text
        .replace('습니다."', '어요."')
        .replace('입니다."', '어요."')
        .replace('습니다', '다')
        .replace('입니다', '다')
    )
    return replaced_text

def count_tokens(model, prompt):
    response = model.count_tokens(prompt)
    total_tokens = response.total_tokens
    return total_tokens

for i in tqdm(range(n), desc="Generating Text"):
    print(f"*** 현재 {i+1}번째 반복 중 ***")

    while True:
        if i == 0:
            current_prompt = prompt_sum + prompt_action
        elif i == n - 1:
            current_prompt = prompt_genre + prompt_char + prompt_back + prompt_rule + prompt_end + prompt
        else:
            current_prompt = prompt_sum + prompt

        response = model.generate_content(current_prompt, generation_config=generation_config)

        try:
            if response.candidates or response.parts or response.parts[0]:
                generated_text = response.parts[0].text
            else:
                continue

            sentences = [sentence.strip() for sentence in response.parts[0].text.split('.') if sentence.strip()]
            exclude_empty_string = True
            duplicate_sentences_internal = [sentence for sentence in sentences if sentences.count(sentence) > 1]
            duplicate_sentences_external = [
                sentence
                for sentence in sentences
                if (
                    sentence in myNovel
                    and (not exclude_empty_string or sentence != '')
                    and all(keyword not in sentence for keyword in keyword_exclude)
                )
            ]
            if duplicate_sentences_internal or duplicate_sentences_external:
                continue

            for keyword in noise_except:
                pattern = re.compile(r'\d+' + re.escape(keyword))
                not_noise = re.findall(pattern, generated_text)
            noise_characters = re.findall("[^가-힣\s\n!.,?~()“”‘’…''\"']", generated_text)
            if noise_characters:
                filtered_noise = [char for char in noise_characters if char not in not_noise]
                continue

            if any(keyword in generated_text for keyword in keyword_banned) and i != n - 1:
                sentences_with_keyword = [
                    sentence.strip()
                    for sentence in generated_text.split('.')
                    if any(keyword in sentence for keyword in keyword_banned)
                ]
                for sentence in sentences_with_keyword:
                    continue

        except (IndexError, AttributeError, ValueError) as e:
            continue

        myNovel += generated_text + '\n\n'
        remaining_tokens = 20000 - count_tokens(model, prompt_sum)
        remaining_tokens = remaining_tokens // 25
        prompt = myNovel.split('.')
        prompt = '.'.join(prompt[-remaining_tokens:]).strip()
        break

myNovel = replace_strings(myNovel)
print(myNovel)


# 파일을 저장합니다. 드라이브 마운트를 해주세요.
from datetime import datetime
import os

# 드라이브 마운트하고 경로를 설정하세요.
# 로컬버전에서는 코드 밑에 data폴더를 만들어 디렉토리 패쓰로 저장. 
# directory_path = "/content/drive/MyDrive/novel/"
directory_path = "novel/"

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join(directory_path, f"{current_time}_novel.txt")
combined_prompt_path = os.path.join(directory_path, f"{current_time}_prompt.txt")

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

with open(file_path, "w", encoding="utf-8") as novel_file:
    novel_file.write(myNovel)

with open(combined_prompt_path, "w", encoding="utf-8") as prompt_file:
    prompt_file.write(f"Sum of Token: {total_tokens_accumulated}\n")
    prompt_file.write("\nPrompt Title:\n")
    prompt_file.write(prompt_genre + "\n")
    prompt_file.write("\nPrompt Rule:\n")
    prompt_file.write(prompt_rule + "\n")
    prompt_file.write("\nPrompt Char:\n")
    prompt_file.write(prompt_char + "\n")
    prompt_file.write("\nPrompt Back:\n")
    prompt_file.write(prompt_back + "\n")
    prompt_file.write("\nPrompt Start:\n")
    prompt_file.write(prompt_start + "\n")
    prompt_file.write("\nPrompt End:\n")
    prompt_file.write(prompt_end + "\n")

print(f"Novel 파일 저장완료: {file_path}")
print(f"Prompt 파일 저장완료: {combined_prompt_path}")

## 1Q님 코드 삽입 끝





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
