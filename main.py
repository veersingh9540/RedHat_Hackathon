import argparse
from data_processor import DataProcessor
from resume_matcher_model import ResumeMatcherModel
from utils import pdf_to_text 

def read_text_file(file_path):
    """Read the contents of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()  
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None

def train_model(file_path):
    data_processor = DataProcessor(file_path)
    data = data_processor.load_and_clean_data()
    
    vectorized_resumes = data_processor.vectorize_data(data['Resume'])
    labels = data['Category']
    
    resume_matcher_model = ResumeMatcherModel(vectorizer=data_processor.vectorizer)
    resume_matcher_model.train(vectorized_resumes, labels)
    
    resume_matcher_model.evaluate(vectorized_resumes, labels)
    
    resume_matcher_model.save_model()

def predict_category(resume_path, jd_path, model_path="resume_matcher_model.pkl", vectorizer_path="vectorizer.pkl"):

    resume_text = pdf_to_text(resume_path)
    
    job_description_text = read_text_file(jd_path)


    if resume_text is None:
        print("Error: Resume text could not be extracted. Please check the PDF file.")
        return
    if job_description_text is None:
        print("Error: Job description text could not be extracted. Please check the TXT file.")
        return

    resume_matcher_model = ResumeMatcherModel(vectorizer=None)  
    resume_matcher_model.load_model(model_path, vectorizer_path)
    
    category = resume_matcher_model.predict_category(resume_text)
    print(f"Predicted Category: {category[0]}")

    match_percentage = resume_matcher_model.calculate_match_percentage(resume_text, job_description_text)
    print(f"Match Percentage: {match_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict using Resume Matcher")
    parser.add_argument("--train", help="Path to the training CSV file", required=False)
    parser.add_argument("--predict", help="Path to the resume PDF file for prediction", required=False)
    parser.add_argument("--jd", help="Path to the job description TXT file", required=False)  
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.train)
    elif args.predict and args.jd:  
        predict_category(args.predict, args.jd)
    else:
        print("Please provide either --train or both --predict and --jd arguments.")
