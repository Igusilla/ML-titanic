import argparse
import numpy as np
import pandas as pd
import joblib

TITLE_MAP = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5
}

EMBARKED_MAP = {'C': 0, 'Q': 1, 'S': 2}
SEX_MAP = {'female': 0, 'male': 1}
AGE_BAND_MAP = {'Child': 0, 'Teen': 1, 'YoungAdult': 2, 'Adult': 3, 'Senior': 4}


def get_title_from_sex_age(sex, age):
    "Estimate title from sex and age"
    if sex == 'male':
        return 'Master' if age < 15 else 'Mr'
    else:
        return 'Miss' if age < 18 else 'Mrs'


def get_age_band(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 35:
        return 'YoungAdult'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'


def build_features(pclass, sex, age, sibsp, parch, fare, embarked):
    family_size = sibsp + parch + 1
    is_alone = int(family_size == 1)
    log_fare = np.log1p(fare)
    title = get_title_from_sex_age(sex, age)
    age_band = get_age_band(age)

    features = {
        'Pclass': pclass,
        'Sex_encoded': SEX_MAP[sex.lower()],
        'Age': age,
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'LogFare': log_fare,
        'Embarked_encoded': EMBARKED_MAP[embarked.upper()],
        'Title_encoded': TITLE_MAP[title],
        'AgeBand_encoded': AGE_BAND_MAP[age_band]
    }

    return pd.DataFrame([features])


def predict(args):
    model = joblib.load('models/best_model_rf.pkl')

    X = build_features(
        pclass=args.pclass,
        sex=args.sex,
        age=args.age,
        sibsp=args.sibsp,
        parch=args.parch,
        fare=args.fare,
        embarked=args.embarked
    )

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    print('\nTITANIC SURVIVAL PREDICTOR')
    print(f'Passenger Class : {args.pclass}')
    print(f'Sex             : {args.sex}')
    print(f'Age             : {args.age}')
    print(f'Family Size     : {args.sibsp + args.parch + 1}')
    print(f'Fare            : £{args.fare}')
    print(f'Embarked        : {args.embarked}')
    print('----------------------------------------------')
    print(f'Prediction      : {"SURVIVED" if prediction == 1 else "DID NOT SURVIVE"}')
    print(f'Survival Prob   : {probability[1]*100:.1f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Titanic survival')
    parser.add_argument('--pclass', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--sex', type=str, default='male', choices=['male', 'female'])
    parser.add_argument('--age', type=float, default=30)
    parser.add_argument('--sibsp', type=int, default=0)
    parser.add_argument('--parch', type=int, default=0)
    parser.add_argument('--fare', type=float, default=15)
    parser.add_argument('--embarked', type=str, default='S', choices=['S', 'C', 'Q'])

    args = parser.parse_args()
    predict(args)