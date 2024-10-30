import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def load_and_clean_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.file_path)
        data.dropna(subset=['Category', 'Resume'], inplace=True)
        data['Resume'] = data['Resume'].apply(self._clean_text)
        return data

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def vectorize_data(self, resumes: pd.Series):
        self.vectorizer.fit(resumes)
        return self.vectorizer.transform(resumes)
