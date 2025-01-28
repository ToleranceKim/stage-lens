import streamlit as st
import cv2
import requests
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from io import BytesIO
from gensim.models import FastText
import re
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 설정
KOPIS_API_KEY = os.getenv('KOPIS_API_KEY')

class KopisAPI:
    def __init__(self, service_key):
        self.service_key = service_key
        self.base_url = "http://www.kopis.or.kr/openApi/restful"
    
    def get_performance_list(self, start_date, end_date):
        url = f"{self.base_url}/pblprfr"
        params = {
            'service': self.service_key,
            'stdate': start_date,
            'eddate': end_date,
            'rows': 100,
            'cpage': 1
        }
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        
        performances = []
        for db in root.findall('.//db'):
            perf = {}
            for child in db:
                perf[child.tag] = child.text
            performances.append(perf)
        
        return performances
    
    def get_performance_detail(self, mt20id: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/pblprfr/{mt20id}"
        params = {'service': self.service_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            db = root.find('.//db')
            
            if db is None:
                return None
                
            detail = {}
            for elem in db:
                if elem.tag == 'styurls':
                    urls = []
                    for styurl in elem.findall('styurl'):
                        if styurl.text and styurl.text.strip():
                            urls.append(styurl.text.strip())
                    detail['styurls'] = urls
                else:
                    if elem.text and elem.text.strip():
                        detail[elem.tag] = elem.text.strip()
                    
            return detail
            
        except Exception as e:
            st.error(f"API 요청 오류: {e}")
            return None

class TextProcessor:
    def __init__(self):
        self.model = None
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    
    def enhance_image(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        return img
    
    def preprocess_image(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return processed
    
    def get_image_sections(self, img):
        width, height = img.size
        sections = []
        
        section_height = height // 3
        for i in range(3):
            top = i * section_height
            bottom = (i + 1) * section_height
            section = img.crop((0, top, width, bottom))
            sections.append(section)
        
        return sections
    
    def extract_text_from_image(self, image_url):
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            
            if img.format == 'GIF':
                img = img.convert('RGB')
            
            target_width = 1000
            width_percent = (target_width / float(img.size[0]))
            target_height = int(float(img.size[1]) * float(width_percent))
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            enhanced_img = self.enhance_image(img)
            img_array = np.array(enhanced_img)
            processed_img = self.preprocess_image(img_array)
            sections = self.get_image_sections(img)
            
            texts = []
            
            for section in sections:
                configs = ['--oem 3 --psm 6', '--oem 3 --psm 1', '--oem 3 --psm 4']
                
                section_texts = []
                for config in configs:
                    text = pytesseract.image_to_string(
                        section, 
                        lang='kor+eng',
                        config=config
                    )
                    if text.strip():
                        section_texts.append(text)
                
                if section_texts:
                    longest_text = max(section_texts, key=len)
                    texts.append(longest_text)
            
            processed_text = pytesseract.image_to_string(
                processed_img,
                lang='kor+eng',
                config='--oem 3 --psm 6'
            )
            texts.append(processed_text)
            
            combined_text = ' '.join(texts)
            cleaned_text = self.clean_text(combined_text)
            
            return cleaned_text
            
        except Exception as e:
            st.error(f"이미지 처리 중 오류 발생: {str(e)}")
            return ""
    
    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        words = text.split()
        words = list(dict.fromkeys(words))
        text = ' '.join(words)
        return text.strip().lower()

    def train_model(self, texts):
        texts = [text for text in texts if text.strip()]
        if not texts:
            st.warning("경고: 학습할 텍스트가 없습니다.")
            return
            
        sentences = [[word for word in text.split()] for text in texts]
        try:
            self.model = FastText(
                sentences=sentences, 
                vector_size=100, 
                window=5, 
                min_count=1,
                workers=4
            )
            st.success(f"모델 학습 완료: {len(sentences)} 문장")
        except Exception as e:
            st.error(f"모델 학습 오류: {str(e)}")
    
    def get_text_vector(self, text):
        if self.model is None:
            st.warning("경고: 모델이 학습되지 않았습니다.")
            return np.zeros(100)
            
        words = text.split()
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not word_vectors:
            return np.zeros(100)
        return np.mean(word_vectors, axis=0)

class PerformanceRecommender:
    def __init__(self, api_client, text_processor):
        self.api_client = api_client
        self.text_processor = text_processor
        self.performances_df = None
    
    def collect_performance_data(self, days=30, max_items=10):
        start_date = datetime.now().strftime("%Y%m%d")
        end_date = (datetime.now() + timedelta(days=days)).strftime("%Y%m%d")
        
        with st.spinner('공연 데이터를 수집하고 있습니다...'):
            performances = []
            perf_list = self.api_client.get_performance_list(start_date, end_date)
            
            progress_bar = st.progress(0)
            for i, perf in enumerate(perf_list[:max_items]):
                mt20id = perf['mt20id']
                detail = self.api_client.get_performance_detail(mt20id)
                
                if detail:
                    poster_text = ""
                    if 'poster' in detail and detail['poster'] and detail['poster'].startswith('http'):
                        try:
                            poster_text = self.text_processor.extract_text_from_image(detail['poster'])
                        except Exception as e:
                            st.warning(f"포스터 이미지 처리 오류({mt20id}): {str(e)}")
                    
                    intro_texts = []
                    if 'styurls' in detail and isinstance(detail['styurls'], list):
                        for img_url in detail['styurls']:
                            if img_url and img_url.startswith('http'):
                                try:
                                    text = self.text_processor.extract_text_from_image(img_url)
                                    if text:
                                        intro_texts.append(text)
                                except Exception as e:
                                    st.warning(f"소개이미지 처리 오류({mt20id}): {str(e)}")
                    
                    all_text = ' '.join(filter(None, [poster_text] + intro_texts))
                    
                    performances.append({
                        'mt20id': mt20id,
                        'title': detail.get('prfnm', ''),
                        'plot': all_text if all_text.strip() else ""
                    })
                
                progress_bar.progress((i + 1) / max_items)
            
            self.performances_df = pd.DataFrame(performances)
            return self.performances_df
    
    def prepare_model(self):
        if self.performances_df is None:
            st.error("공연 데이터를 먼저 수집하세요.")
            return
        
        plots = self.performances_df['plot'].tolist()
        self.text_processor.train_model(plots)
    
    def get_recommendations(self, user_plot, top_n=5):
        if self.performances_df is None:
            st.error("공연 데이터를 먼저 수집하세요.")
            return
        
        user_vector = self.text_processor.get_text_vector(user_plot)
        
        similarities = []
        for plot in self.performances_df['plot']:
            plot_vector = self.text_processor.get_text_vector(plot)
            similarity = np.dot(user_vector, plot_vector) / (
                np.linalg.norm(user_vector) * np.linalg.norm(plot_vector)
            )
            similarities.append(similarity)
        
        self.performances_df['similarity'] = similarities
        recommendations = self.performances_df.nlargest(top_n, 'similarity')
        return recommendations[['title', 'similarity']]

def main():
    st.set_page_config(
        page_title="Stage Lense - KOPIS 공연 추천 시스템",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("Stage Lense - KOPIS 공연 추천 시스템 🎭")
    
    # API 클라이언트 초기화
    api_client = KopisAPI(KOPIS_API_KEY)
    text_processor = TextProcessor()
    recommender = PerformanceRecommender(api_client, text_processor)
    
    # 데이터 수집
    if 'data_collected' not in st.session_state:
        st.session_state.data_collected = False
    
    if not st.session_state.data_collected:
        days = st.slider("데이터 수집 기간 (일)", 1, 31, 7)
        max_items = st.slider("수집할 공연 수", 5, 100, 10)
        
        if st.button("데이터 수집 시작"):
            performances_df = recommender.collect_performance_data(days=days, max_items=max_items)
            recommender.prepare_model()
            st.session_state.data_collected = True
            st.session_state.recommender = recommender
            st.success("데이터 수집 및 모델 학습이 완료되었습니다!")
    
    # 공연 추천
    if st.session_state.data_collected:
        st.subheader("공연 추천")
        
        user_input = st.text_area("보고 싶은 이야기나 키워드를 입력해주세요:", 
                                 height=100,
                                 placeholder="예: 사랑과 우정, 판타지 모험, 역사적 사건 등")
        
        if st.button("공연 추천받기"):
            if user_input:
                with st.spinner("추천 공연을 찾고 있습니다..."):
                    recommendations = st.session_state.recommender.get_recommendations(user_input)
                    
                    st.subheader("추천 공연 목록")
                    for _, row in recommendations.iterrows():
                        score = int(row['similarity'] * 100)
                        st.markdown(f"#### {row['title']}")
                        st.progress(score)
                        st.markdown(f"일치도: {score}%")
                        st.divider()
            else:
                st.warning("이야기나 키워드를 입력해주세요.")

if __name__ == "__main__":
    main()
