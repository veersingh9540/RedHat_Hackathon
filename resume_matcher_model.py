import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

class ResumeMatcherModel:
    def __init__(self, vectorizer):
        self.model = LogisticRegression(max_iter=1000)  
        self.vectorizer = vectorizer  

    def train(self, features, labels):
        self.model.fit(features, labels)
    
    def evaluate(self, features, labels):
        predictions = self.model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
    def save_model(self, model_path="resume_matcher_model.pkl", vectorizer_path="vectorizer.pkl"):

        with open(model_path, "wb") as model_file, open(vectorizer_path, "wb") as vectorizer_file:
            pickle.dump(self.model, model_file)
            pickle.dump(self.vectorizer, vectorizer_file)

    def load_model(self, model_path="resume_matcher_model.pkl", vectorizer_path="vectorizer.pkl"):

        with open(model_path, "rb") as model_file, open(vectorizer_path, "rb") as vectorizer_file:
            self.model = pickle.load(model_file)
            self.vectorizer = pickle.load(vectorizer_file)

    def predict_category(self, resume_text):
        resume_vector = self.vectorizer.transform([resume_text])
        return self.model.predict(resume_vector)
           
    def calculate_match_percentage(self, resume_text, job_description_text):

        resume_vector = self.vectorizer.transform([resume_text])
        job_description_vector = self.vectorizer.transform([job_description_text])
        

        similarity = cosine_similarity(resume_vector, job_description_vector)
        
        match_percentage =  similarity[0][0] * 100  + 50 # Buffer for personality and Extracurricular 
        return match_percentage
