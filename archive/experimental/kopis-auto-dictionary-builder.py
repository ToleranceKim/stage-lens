# auto_dictionary_builder.py

from datetime import datetime, timedelta
from collections import Counter
from typing import Set, Dict, List
import re
from textrank import KeywordSummarizer

class AutoDictionaryBuilder:
    """
    공연 도메인을 위한 자동 사전 구축 도구
    
    이 클래스는 다음 세 가지 주요 기능을 제공합니다:
    1. KOPIS API 응답으로부터 도메인 용어 자동 추출
    2. OCR 결과에서 발견되는 오류 패턴을 분석하여 교정 사전 구축
    3. TextRank 알고리즘을 사용한 핵심 키워드 추출
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.domain_terms = set()
        self.corrections = {}
        self.ocr_results = []
        
    def build_domain_dictionary(self, api_client) -> Set[str]:
        """
        KOPIS API 응답을 분석하여 도메인 용어 사전을 구축합니다.
        
        Args:
            api_client: KOPIS API 클라이언트 인스턴스
            
        Returns:
            Set[str]: 추출된 도메인 용어 집합
        """
        try:
            # 1년치 공연 데이터 수집
            start_date = datetime.now().strftime("%Y%m%d")
            end_date = (datetime.now() + timedelta(days=365)).strftime("%Y%m%d")
            performances = api_client.get_performance_list(start_date, end_date)
            
            domain_terms = set()
            
            # 공연 데이터에서 주요 필드 추출
            for perf in performances:
                # 기본 정보 추출
                fields_to_extract = ['prfnm', 'genrenm', 'fcltynm']
                for field in fields_to_extract:
                    if field in perf:
                        domain_terms.add(perf[field])
                
                # 상세 정보 수집
                if 'mt20id' in perf:
                    detail = api_client.get_performance_detail(perf['mt20id'])
                    if detail:
                        # 출연진, 제작진 정보 추출
                        for field in ['prfcast', 'prfcrew']:
                            if field in detail and detail[field]:
                                terms = detail[field].split(',')
                                domain_terms.update(term.strip() for term in terms)
            
            # 용어 정제
            domain_terms = {
                term.strip() for term in domain_terms 
                if len(term.strip()) > 1  # 1글자 용어 제외
            }
            
            self.domain_terms = domain_terms
            return domain_terms
            
        except Exception as e:
            print(f"도메인 사전 구축 중 오류: {str(e)}")
            return set()
    
    def build_correction_dictionary(self) -> Dict[str, str]:
        """
        OCR 오류 패턴을 기반으로 교정 사전을 구축합니다.
        
        Returns:
            Dict[str, str]: {잘못된 형태: 올바른 형태} 매핑
        """
        # 기본 교정 규칙
        base_corrections = {
            '처옴': '처음',
            '그늘': '그날',
            '연줄': '연출',
            '공언': '공연',
            '배우들': '배우들'
        }
        
        # 한글 자음/모음과 비슷하게 생긴 문자들의 매핑
        similar_chars = {
            'ㅇ': ['o', 'O', '0'],
            'ㄹ': ['2', 'Z', 'z'],
            'ㅂ': ['B', 'b'],
            'ㅎ': ['H', 'h'],
            'ㅅ': ['A', 'a'],
            'ㅋ': ['k', 'K'],
            'ㅌ': ['t', 'T'],
            'ㅊ': ['c', 'C']
        }
        
        corrections = base_corrections.copy()
        
        # 도메인 용어에 대한 자동 교정 규칙 생성
        for term in self.domain_terms:
            for char in term:
                if char in similar_chars:
                    for wrong_char in similar_chars[char]:
                        wrong_term = term.replace(char, wrong_char)
                        corrections[wrong_term] = term
        
        self.corrections = corrections
        return corrections
    
    def add_ocr_result(self, text: str) -> None:
        """
        OCR 결과를 저장하여 추후 분석에 활용합니다.
        
        Args:
            text: OCR로 추출된 텍스트
        """
        if text and isinstance(text, str):
            self.ocr_results.append(text)
    
    def analyze_ocr_patterns(self) -> Dict[str, str]:
        """
        저장된 OCR 결과들을 분석하여 추가 교정 패턴을 찾습니다.
        
        Returns:
            Dict[str, str]: 발견된 추가 교정 패턴
        """
        if not self.ocr_results:
            return {}
            
        word_pairs = []
        
        # 비슷한 단어들을 쌍으로 수집
        for text in self.ocr_results:
            words = text.split()
            for i in range(len(words)-1):
                if self._are_similar_words(words[i], words[i+1]):
                    word_pairs.append((words[i], words[i+1]))
        
        # 빈도 분석으로 올바른 형태 추정
        corrections = {}
        word_counts = Counter(w for pair in word_pairs for w in pair)
        
        for wrong, right in word_pairs:
            if word_counts[wrong] < word_counts[right]:
                corrections[wrong] = right
        
        return corrections
    
    def extract_keywords(self, top_n: int = 100) -> Set[str]:
        """
        TextRank 알고리즘으로 OCR 결과에서 주요 키워드를 추출합니다.
        
        Args:
            top_n: 추출할 키워드 수
            
        Returns:
            Set[str]: 추출된 키워드 집합
        """
        if not self.ocr_results:
            return set()
            
        summarizer = KeywordSummarizer(
            tokenize=self.tokenizer.morphs,
            window=-1,
            verbose=False
        )
        
        # 모든 OCR 결과를 하나의 문서로 결합
        combined_text = ' '.join(self.ocr_results)
        
        try:
            keywords = summarizer.summarize(combined_text, top_n)
            return {word for word, rank in keywords}
        except Exception as e:
            print(f"키워드 추출 중 오류: {str(e)}")
            return set()
    
    def _are_similar_words(self, word1: str, word2: str) -> bool:
        """
        두 단어가 비슷한지 확인합니다 (편집 거리 사용).
        """
        if abs(len(word1) - len(word2)) > 2:
            return False
            
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # 단어 길이의 30% 이하의 편집 거리를 허용
        max_distance = max(len(word1), len(word2)) * 0.3
        return levenshtein_distance(word1, word2) <= max_distance

def usage_example():
    """사용 예시"""
    from konlpy.tag import Okt
    
    # 초기화
    tokenizer = Okt()
    builder = AutoDictionaryBuilder(tokenizer)
    
    # KOPIS API로부터 도메인 사전 구축
    api_client = KopisAPI("your-api-key")
    domain_terms = builder.build_domain_dictionary(api_client)
    print(f"추출된 도메인 용어: {len(domain_terms)}개")
    
    # OCR 결과 추가
    builder.add_ocr_result("첫 번째 OCR 결과...")
    builder.add_ocr_result("두 번째 OCR 결과...")
    
    # 교정 사전 구축
    corrections = builder.build_correction_dictionary()
    print(f"생성된 교정 규칙: {len(corrections)}개")
    
    # OCR 패턴 분석
    additional_corrections = builder.analyze_ocr_patterns()
    print(f"발견된 추가 교정 패턴: {len(additional_corrections)}개")
    
    # 키워드 추출
    keywords = builder.extract_keywords(top_n=50)
    print(f"추출된 주요 키워드: {len(keywords)}개")

if __name__ == "__main__":
    usage_example()