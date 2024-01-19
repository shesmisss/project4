from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, TextAreaField
from wtforms.validators import DataRequired


app = Flask(__name__)
app.config['SECRET_KEY'] = 'YourSecretKey'


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
        characters = form.characters.data
        setting = form.setting.data
        genre = form.genre.data
        dialect = form.dialect.data
        novel = generate_novel(characters, setting, genre, dialect)
    return render_template('page1.html', form=form, novel=novel)


def generate_novel(characters, setting, genre, dialect):
    # LangChain과 LLM을 사용하여 소설을 생성하는 로직 구현
    return f"생성된소설은 {characters}, {setting}, {genre}, {dialect}"







@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')




if __name__ == '__main__':
    app.run(debug=True)
