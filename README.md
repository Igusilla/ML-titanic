# Titanic Survival Prediction
A complete machine learning pipeline to predict passenger survival on the Titanic, built with Python and scikit-learn.

## Results
| Model | CV Accuracy (5-fold) |
|---|---|
| Logistic Regression | ~80% |
| Random Forest (tuned) | ~83.5% |
| XGBoost | ~84.1% |

## Dataset from Kaggle Titanic Competition
https://www.kaggle.com/competitions/titanic/data

## Usage
Use the CLI in folder to make predictions:

```bash
python predict.py --pclass 1 --sex female --age 29 --sibsp 0 --parch 0 --fare 100 --embarked S
```

Output:
```
TITANIC SURVIVAL PREDICTOR
Passenger Class : 1
Sex             : female
Age             : 29
Family Size     : 1
Fare            : £100
Embarked        : S
----------------------------------------------
Prediction      : SURVIVED
Survival Prob   : 91.7%
```

## Features Engineered
- **Title** extracted from passenger name (Mr, Mrs, Miss, Master, Rare)
- **FamilySize** = SibSp + Parch + 1
- **IsAlone** flag for solo travelers
- **AgeBand** categorical age groups
- **LogFare** log-transformed fare to reduce skew
- **Smart missing value imputation** per class/sex group

## Key Findings
- Women had ~75% survival rate vs ~20% for men
- 1st class passengers had ~63% survival rate vs ~24% for 3rd class
- Children under 12 had highest survival rates
- Passengers traveling alone had lower survival rates than those with small families

## Tech Stack
Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn
