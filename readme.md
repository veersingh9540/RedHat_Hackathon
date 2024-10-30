# Resume Matcher Project

This project implements a Resume Matcher that predicts the category of a resume based on its content and evaluates how well it matches a given job description using cosine similarity.

## Table of Contents

- [Requirements]
- [Installation]
- [Usage](#usage)
- [How the Prediction Model Works]
- [Cosine Similarity](#cosine-similarity)

## Requirements

- Python 3.6 or higher
- Libraries:
  - `scikit-learn`
  - `joblib`
  - `PyMuPDF` (for PDF processing)


## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. To Train the Model and then predict on the resume and JD
   use Path to JD and resume
   ```bash
   python main.py --train UpdatedResumeDataSet.csv  
   python main.py --predict Resume/resume3.pdf --jd jd.txt
   

