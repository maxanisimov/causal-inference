"""
Estimating the causal effect of treatment on blood pressure in a simulated example
adapted from Luque-Fernandez et al. (2018):
    https://academic.oup.com/ije/article/48/2/640/5248195
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import os
import sys
my_modules_path = os.path.join('/Users/maksimanisimov/Documents/repos/maksim-anisimov', 'my-modules')
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, my_modules_path)
import simulations_utils
# import cl_utils
import matplotlib.pyplot as plt

# this is a confounder that causes two series: treatment and target

def generate_ts_data(confounder_ar_coefs=[0.6], confounder_ma_coefs=[],
                     nsample=1000, confounder_noise_scale=1,
                     beta1=0.5, alpha1=0.4, alpha2=0.3,
                     random_seed=0, binary_treatment=False, binary_cutoff=0):
    confounder_process = simulations_utils.simulate_arma_process(ar_coefs=confounder_ar_coefs,
                                                                 ma_coefs=confounder_ma_coefs,
                                                                 nsample=nsample,
                                                                 scale=confounder_noise_scale,
                                                                 random_seed=random_seed)
    np.random.seed(random_seed)
    treatment_process = beta1*confounder_process + np.random.normal(size=nsample)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = treatment_process.mean()
        treatment_process = (treatment_process > binary_cutoff).astype(int)
    # target # is caused by both confounder and treatment
    target_process = alpha1*confounder_process + alpha2*treatment_process + np.random.normal(size=nsample)

    return pd.DataFrame({'target': target_process, 'treatment': treatment_process,
                         'confounder': confounder_process})

def estimate_causal_effect(Xt, y, model=LinearRegression(), treatment_idx=0, regression_coef=False):
    model.fit(Xt, y)
    if regression_coef:
        return model.coef_[treatment_idx]
    else:
        Xt1 = pd.DataFrame.copy(Xt)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0 = pd.DataFrame.copy(Xt)
        Xt0[Xt.columns[treatment_idx]] = 0
        return (model.predict(Xt1) - model.predict(Xt0)).mean()

"""
The ATE estimate is better when coefficients are largers - why? Try (alpha1=.4, alpha2=.3) vs (alpha1=.8, alpha2=.7)
"""
if __name__ == '__main__':
    binary_t_df = generate_ts_data(beta1=.5, alpha1=.8, alpha2=.7, binary_treatment=True)
    continuous_t_df = generate_ts_data(beta1=.5, alpha1=.8, alpha2=.7, binary_treatment=False)

    ate_est_naive = None
    ate_est_adjust_all = None
    ate_est_adjust_confounder = None

    for df, name in zip([binary_t_df, continuous_t_df],
                        ['Binary Treatment Data', 'Continuous Treatment Data']):
        print()
        print(f'### {name} ###')
        print()

        # Adjustment formula estimates
        ate_est_naive = estimate_causal_effect(df[['treatment']], df['target'], treatment_idx=0)
        # ate_est_adjust_all = estimate_causal_effect(df[['treatment', 'confounder', 'proteinuria']],
        #                                             df['target'], treatment_idx=0)
        ate_est_adjust_confounder = estimate_causal_effect(df[['treatment', 'confounder']], df['target'])
        print('# Adjustment Formula Estimates #')
        print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
        # print('ATE estimate adjusting for all covariates:\t', ate_est_adjust_all)
        print('ATE estimate adjusting for confounder:\t\t\t\t', ate_est_adjust_confounder)
        print()

        # Linear regression coefficient estimates
        ate_est_naive = estimate_causal_effect(df[['treatment']], df['target'], treatment_idx=0,
                                               regression_coef=True)
        # ate_est_adjust_all = estimate_causal_effect(df[['treatment', 'confounder', 'proteinuria']],
        #                                             df['target'], treatment_idx=0,
        #                                             regression_coef=True)
        ate_est_adjust_confounder = estimate_causal_effect(df[['treatment', 'confounder']], df['target'],
                                                    regression_coef=True)
        print('# Regression Coefficient Estimates #')
        print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
        # print('ATE estimate adjusting for all covariates:\t', ate_est_adjust_all)
        print('ATE estimate adjusting for confounder:\t\t\t\t', ate_est_adjust_confounder)
        print()


"""
Now, the idea is to AUTOMATICALLY FIND CONFOUNDERS and include them in feature list to build a model
thereafter
"""