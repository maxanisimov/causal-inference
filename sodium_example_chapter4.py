import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_data(n=1000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3, binary_treatment=True, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)
    proteinuria = alpha1 * sodium + alpha2 * blood_pressure + np.random.normal(size=n)
    hypertension = (blood_pressure >= 140).astype(int)  # not used, but could be used for binary outcomes
    return pd.DataFrame({'blood_pressure': blood_pressure, 'sodium': sodium,
                         'age': age, 'proteinuria': proteinuria})


if __name__ == '__main__':
    treatment_effect = 1.05
    df = generate_data(beta1=treatment_effect)

    Xt = df[['sodium', 'age']]  # take only a treatment and confounders
    y = df['blood_pressure']
    model = LinearRegression()
    model.fit(Xt, y)  # fit the model using real-world data

    Xt1 = pd.DataFrame.copy(Xt)
    Xt1['sodium'] = 1  # do intervention and set sodium to 1 for all individuals (dummy for substantial amount)
    Xt0 = pd.DataFrame.copy(Xt)
    Xt0['sodium'] = 0  # intervene by setting unsubstantial amount for each individual

    ate_est = np.mean(model.predict(Xt1) - model.predict(Xt0))
    print(f'ATE estimate: {ate_est}')
    print(f'True treatment effect: {treatment_effect}')
    print(f'Error: {np.abs(ate_est / treatment_effect - 1)}')
