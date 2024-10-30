from data_processor import DataProcessor
from resume_matcher_model import ResumeMatcherModel
from utils import pdf_to_text


class ResumeMatcher:
    def __init__(self, model: ResumeMatcherModel, processor: DataProcessor):
        self.model = model
        self.processor = processor
        self.model.load_model()

    def predict_match_score(self, resume_pdf_path: str, job_description: str) -> float:
        resume_text = pdf_to_text(resume_pdf_path)
        resume_vector = self.processor.vectorizer.transform([resume_text])
        job_description_vector = self.processor.vectorizer.transform([self.processor._clean_text(job_description)])
        
        if self.model.model is not None:
            match_score = self.model.model.predict_proba(abs(resume_vector - job_description_vector))[0, 1]
            return match_score
        else:
            raise ValueError("Model is not loaded. Please train or load the model first.")
