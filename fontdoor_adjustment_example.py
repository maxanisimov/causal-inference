import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Now assume there is a mediator that carries the causal effect of sodium on blood pressure
# Age is a confounder and effects of all other variables are negligible.
def generate_data(n=1000, seed=0, mediator_coef=1.05, sodium_coef=0.8,
                  binary_treatment=True, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    sodium_mediator = sodium_coef * sodium + np.random.normal(size=n)
    blood_pressure = mediator_coef * sodium_mediator + 2 * age + np.random.normal(size=n)
    # proteinuria = alpha1 * sodium + alpha2 * blood_pressure + np.random.normal(size=n)
    # hypertension = (blood_pressure >= 140).astype(int)  # not used, but could be used for binary outcomes
    return pd.DataFrame({'blood_pressure': blood_pressure, 'sodium': sodium,
                         'sodium_mediator': sodium_mediator, 'age': age})


def estimate_causal_effect(Xt, y, model=LinearRegression(),
                           treatment_colname='sodium',
                           mediator_colname='sodium_mediator',
                           step_two_adjustment_variable='age',
                           regression_coef=True):
    # 1. Estimate a causal effect of sodium intake on its mediator
    # Because Y is a collider on the backdoor path from sodium to the mediator, all backdoor paths are blocked.
    # Hence, the only association that flows from sodium intake to M is causal and we can use
    # a standard statistical estimation.

    treatment_mediator_model = model
    treatment_mediator_model.fit(Xt[[treatment_colname]], Xt[mediator_colname])
    if regression_coef:
        treatment_coef = treatment_mediator_model.coef_[0]
    else:
        pass
        # Xt1 = pd.DataFrame.copy()  # create a dataframe with treatment values of 1
        # Xt1[Xt.columns[treatment_idx]] = 1
        # Xt0 = pd.DataFrame.copy(Xt)  # # create a dataframe with treatment values of 0
        # Xt0[Xt.columns[treatment_idx]] = 0
        # return (model.predict(Xt1) - model.predict(Xt0)).mean()

    # 2. Estimate a causal effect of the mediator on blood pressure
    # Sodium intake blocks the only backdoor path from the mediator to the outcome.
    # Hence, we only need to adjust for it to estimate the causal effect.
    # But, it adjusting for age gives more accurate results.
    mediator_outcome_model = model
    mediator_outcome_model.fit(Xt[[step_two_adjustment_variable, mediator_colname]], y)
    if regression_coef:
        mediator_coef = mediator_outcome_model.coef_[1]
    else:
        pass
        # Xt1 = pd.DataFrame.copy(Xt)  # create a dataframe with treatment values of 1
        # Xt1[Xt.columns[treatment_idx]] = 1
        # Xt0 = pd.DataFrame.copy(Xt)  # # create a dataframe with treatment values of 0
        # Xt0[Xt.columns[treatment_idx]] = 0
        # return (model.predict(Xt1) - model.predict(Xt0)).mean()

    return treatment_coef * mediator_coef


if __name__ == '__main__':
    mediator_coef, sodium_coef = 1.05, 0.8
    fit_intercept = False
    data = generate_data(n=1000, seed=0, mediator_coef=mediator_coef, sodium_coef=sodium_coef,
                         binary_treatment=False)
    Xt = data[['sodium', 'sodium_mediator', 'age']]
    y = data['blood_pressure']
    print(f'True causal coefficient: {mediator_coef*sodium_coef}')

    est_causal_coef = estimate_causal_effect(Xt, y,
                                             model=LinearRegression(fit_intercept=fit_intercept),
                                             treatment_colname='sodium',
                                             mediator_colname='sodium_mediator',
                                             step_two_adjustment_variable='age',
                                             regression_coef=True)
    print(f'Causal coefficient estimated using frontdoor adjustment: {est_causal_coef}')


    ### Non-causal models
    print('Corr matrix')
    print(data.corr())
    # Pairwise non-causal model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(Xt[['sodium']], y)
    pairwise_model_coef = model.coef_[0]
    print(f'Pairwise model coefficient: {pairwise_model_coef}')

    # All-data non-causal model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(Xt, y)
    alldata_model_coef = model.coef_[0]
    print(f'All-data model coefficient: {alldata_model_coef}')

