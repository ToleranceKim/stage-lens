# requirements.txt
Django==4.2.0
requests==2.31.0
scikit-learn==1.3.0
python-dotenv==1.0.0
numpy==1.24.3

# config/settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = 'your-secret-key'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'recommendation',
]

# recommendation/models.py
from django.db import models

class Performance(models.Model):
    mt20id = models.CharField(max_length=20, primary_key=True)
    prfnm = models.CharField(max_length=200)
    genrenm = models.CharField(max_length=100)
    prfpdfrom = models.DateField()
    prfpdto = models.DateField()
    fcltynm = models.CharField(max_length=200)
    poster = models.URLField()
    sty = models.TextField()

# recommendation/views.py
import os
import requests
from datetime import datetime, timedelta
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Performance

def get_performances():
    api_key = os.getenv('KOPIS_API_KEY')
    today = datetime.now()
    start_date = today.strftime('%Y%m%d')
    end_date = (today + timedelta(days=30)).strftime('%Y%m%d')
    
    url = f"http://www.kopis.or.kr/openApi/restful/pblprfr?service={api_key}&stdate={start_date}&eddate={end_date}&rows=100"
    response = requests.get(url)
    performances = []
    
    if response.status_code == 200:
        # XML 파싱 및 공연 정보 추출 로직
        for performance in response.find_all('db'):
            mt20id = performance.find('mt20id').text
            # 상세 정보 API 호출하여 줄거리 정보 가져오기
            detail_url = f"http://www.kopis.or.kr/openApi/restful/pblprfr/{mt20id}?service={api_key}"
            detail_response = requests.get(detail_url)
            if detail_response.status_code == 200:
                performances.append({
                    'mt20id': mt20id,
                    'prfnm': performance.find('prfnm').text,
                    'sty': detail_response.find('sty').text,
                    # 기타 필요한 정보들
                })
    return performances

def find_similar_performances(user_plot):
    performances = get_performances()
    
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer()
    plots = [p['sty'] for p in performances]
    plots.append(user_plot)
    tfidf_matrix = tfidf.fit_transform(plots)
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # 유사도가 높은 순으로 정렬
    similar_indices = cosine_sim.argsort()[::-1][:5]
    recommendations = [performances[i] for i in similar_indices]
    
    return recommendations

def recommend_view(request):
    if request.method == 'POST':
        user_plot = request.POST.get('plot', '')
        recommendations = find_similar_performances(user_plot)
        return render(request, 'recommendation/results.html', {
            'recommendations': recommendations
        })
    return render(request, 'recommendation/input.html')

# recommendation/templates/recommendation/input.html
<!DOCTYPE html>
<html>
<head>
    <title>공연 추천 서비스</title>
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: 'Arial', sans-serif;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            width: 80%;
            max-width: 800px;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        textarea {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            margin-top: 0.5rem;
        }
        button {
            background: #4c1d95;
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            background: #5b21b6;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 공연 추천 서비스</h1>
        <p>관심 있는 줄거리를 입력해주세요. 비슷한 내용의 공연을 추천해드립니다!</p>
        <form method="POST">
            {% csrf_token %}
            <div class="form-group">
                <label for="plot">줄거리:</label>
                <textarea name="plot" id="plot" rows="6" required></textarea>
            </div>
            <button type="submit">추천 받기</button>
        </form>
    </div>
</body>
</html>

# recommendation/templates/recommendation/results.html
<!DOCTYPE html>
<html>
<head>
    <title>추천 결과</title>
    <style>
        /* input.html의 스타일을 기본으로 하되 결과 표시를 위한 추가 스타일 */
        .recommendations {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        .performance-card {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 1rem;
            transition: transform 0.3s ease;
        }
        .performance-card:hover {
            transform: translateY(-5px);
        }
        .performance-card img {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 추천 공연</h1>
        <div class="recommendations">
            {% for performance in recommendations %}
            <div class="performance-card">
                <img src="{{ performance.poster }}" alt="{{ performance.prfnm }}">
                <h3>{{ performance.prfnm }}</h3>
                <p>{{ performance.genrenm }}</p>
                <p>{{ performance.fcltynm }}</p>
                <p>{{ performance.prfpdfrom }} ~ {{ performance.prfpdto }}</p>
            </div>
            {% endfor %}
        </div>
        <a href="{% url 'recommend' %}" class="button">다시 검색하기</a>
    </div>
</body>
</html>