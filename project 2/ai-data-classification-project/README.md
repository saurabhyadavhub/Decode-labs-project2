# AI Project 2: Data Classification Using AI

A GitHub-ready beginner project based on the DecodeLabs Project 2 brief. It demonstrates a supervised machine-learning workflow: load and understand a dataset, split it into training and testing sets, train a classification algorithm, and evaluate its performance.

## Project overview

This project uses the built-in **Iris dataset** from scikit-learn and a **K-Nearest Neighbors (KNN)** classifier to predict the species of an iris flower from four measurements.

### Skills demonstrated
- Loading and exploring a dataset
- Separating features (`X`) and target labels (`y`)
- Splitting data into training and testing sets
- Training a supervised classification model
- Evaluating accuracy and a confusion matrix
- Predicting a class for new measurements

## Requirements

- Python 3.9 or newer
- pip

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project from the repository root
python src/main.py
```

The script prints dataset information, test accuracy, a classification report, a confusion matrix, and a sample prediction. It also saves evaluation output in the `outputs/` folder.

## Example custom prediction

Edit the four values in `src/main.py` under `sample_flower`:
- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

Example: `[5.1, 3.5, 1.4, 0.2]`

## Project structure

```text
ai-data-classification-project/
├── src/
│   └── main.py
├── outputs/
│   └── .gitkeep
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The Iris dataset is included with scikit-learn, so no separate download is required. It contains 150 samples, four numeric features, and three flower classes.

## GitHub upload

1. Create a new repository on GitHub, for example `ai-data-classification-project`.
2. Extract this ZIP file.
3. Upload the extracted files, or run the commands below from the project folder:

```bash
git init
git add .
git commit -m "Add AI data classification project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-data-classification-project.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Note

This is a learning project and a sample implementation of the requirements in the supplied project brief. Run the script locally and review the generated results before submitting it.
