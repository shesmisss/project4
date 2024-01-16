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
