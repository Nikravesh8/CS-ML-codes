if __name__ == '__main__':
    import sys
    import pandas as pd
    from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error, r2_score
    import numpy as np
    from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, LeaveOneOut
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # =========================== Inputs  ====================================

    target = 'penetration depth'  # 'penetration depth' or 'IBE'
    included = ['mass_p', 'velocity', 'density_p', 'bulk_modulus_s', 'yield_stress_s']  # optimum features for dp
    # included = ['velocity', 'mass_p', 'Young_modulus_p', 'velocity_of_sound_p', 'bulk_modulus_s', 'velocity_of_sound_s']  # optimum features for IBE

    n_restarts_optimizer = 9
    n_cpu = 2  # -1 for all cpu

    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
    # crossvalidation = LeaveOneOut()

    # =========================== Load and process data  ====================================

    if target == 'penetration depth':
        str_index = 'dp'
    else:
        str_index = 'IBE'

    sys.stdout = open('Output_' + str_index + '.txt', 'w')

    ## Import data

    df_mp0 = pd.read_excel("Dataframe.xlsx", index_col=[0])

    ### Data cleaning

    if target == 'IBE':
        indexAge = df_mp0[(df_mp0['velocity'] == 14) & (df_mp0['particle diameter'] == 16) & (df_mp0['particle material'] == 'Pd') & (df_mp0['substrate material'] == 'Cu')].index
        df_mp0.drop(indexAge, inplace=True)

    df_mp0 = df_mp0[((df_mp0['penetration depth'] < 110) & (df_mp0['particle diameter'] == 8)) |
                    ((df_mp0['penetration depth'] < 40) & (df_mp0['particle diameter'] == 4)) |
                    ((df_mp0['penetration depth'] < 230) & (df_mp0['particle diameter'] == 16))]

    ## Define input data and output data

    y = df_mp0[target].values

    excluded = ['particle material', 'substrate material', 'penetration depth', 'particle composition',
                'substrate composition', 'Ac', 'IBE', 'E0', 'Ef', 'particle diameter']
    X = df_mp0.drop(excluded, axis=1)

    X = X[included]

    print("There are {} descriptors:\n\n{}".format(X.shape[1], X.columns.values))

    # ================= Try some different machine learning models =============================

    ## GPR model

    kernel = 1**2 * RBF() + WhiteKernel()
    gaussian_process = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=False, alpha=0.0, random_state=1))

    # compute cross validation scores for random forest model

    r2_scores = cross_val_score(gaussian_process, X, y, scoring='r2', cv=crossvalidation, n_jobs=n_cpu)
    scores = cross_val_score(gaussian_process, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=n_cpu)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    mae_scores = cross_val_score(gaussian_process, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=n_cpu)

    print('GPR Cross-validation results:')
    print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
    print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))
    print('Folds: %i, mean MAE: %.3f' % (len(scores), -mae_scores.mean()))

    # Plot actual vs predicted

    plt.figure()
    plt.rcParams['font.size'] = '12'
    y_pred = cross_val_predict(gaussian_process, X, y, cv=crossvalidation)
    plt.scatter(y, y_pred, c='#FF2E63', alpha=0.8)

    if target == 'penetration depth':
        plt.xlabel("Observed" + ' $h_d$ ' + "from MD simulation (Å)")
        plt.ylabel("Predicted" + ' $h_d$ ' + "(Å)")
    else:
        plt.xlabel("Observed" + ' IBE ' + "from MD simulation (J/${m}^{2}$)")
        plt.ylabel("Predicted" + ' IBE ' + "(J/${m}^{2}$)")

    bound_min = min(y.min(), y_pred.min())
    bound_max = max(y.max(), y_pred.max())
    bounds = (bound_min, bound_max)
    plt.plot(bounds, bounds, 'k-')
    plt.xlim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.ylim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.gca().axes.set_aspect('equal', 'box')
    plt.title('GPR regression')
    plt.show()
    plt.savefig('fig1_' + str_index + '.png', dpi=500)
    pd.DataFrame({"Observed": y, "Predicted": y_pred}).to_excel("Final_Parity_" + str_index + ".xlsx")

    sys.stdout.close()
