# Resume Matcher Project

This project implements a Resume Matcher that predicts the category of a resume based on its content and evaluates how well it matches a given job description using cosine similarity.

## Table of Contents

- [Requirements]
- [Installation]
- [Usage]
- [How the Prediction Model Works]
- [Cosine Similarity]

## Requirements

- Python 3.6 or higher
- Libraries:
  - `scikit-learn`
  - `joblib`
  - `PyMuPDF` (for PDF processing)


## Installation

1. Clone this repository and run requirements.txt:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   pip intall requirements.txt
2. To Train the Model and then predict on the resume and JD
   use Path to JD and resume
   ```bash
   python main.py --train UpdatedResumeDataSet.csv  
   python main.py --predict Resume/resume3.pdf --jd jd.txt

3. Note Virual env is not necessary :
  ```bash
  python -m venv myenv
  source myenv/bin/activate




