if __name__ == '__main__':
    import sys
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, make_scorer
    import numpy as np
    from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, LeaveOneOut
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestRegressor
    from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
    from sklearn.feature_selection import RFECV
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # =========================== Inputs  ====================================

    target = 'penetration depth'  # 'penetration depth' or 'IBE'
    corr_threshold = 0.80
    corr_method = 'pearson'  # 'pearson' or 'spearman'
    n_cpu = 1  # -1 for all cpu
    EFS_max_features = 1
    n_reset_optimizer = 1     # 9

    multicollinear_thresh = 10

    crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)

    # =========================== Functions  ====================================

    # compute the vif for all given features
    def calculate_vif(df, features):
        vif, tolerance = {}, {}
        # all the features that you want to examine
        for feature in features:
            # extract all the other features you will regress against
            X = [f for f in features if f != feature]
            X, y = df[X], df[feature]
            # extract r-squared from the fit
            r2 = LinearRegression().fit(X, y).score(X, y)

            # calculate tolerance
            tolerance[feature] = 1 - r2
            # calculate VIF
            vif[feature] = 1 / (tolerance[feature])
        # return VIF DataFrame
        return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


    def corr_vif_features(corr_features_df, thresh=multicollinear_thresh):
        variables = list(range(corr_features_df.shape[1]))
        to_drop_final = []
        dropped = True
        while dropped:
            dropped = False
            vif = calculate_vif(corr_features_df, corr_features_df.columns[variables])
            print(vif)

            maxloc = np.argmax(vif['VIF'])
            if max(vif['VIF']) > thresh:
                print('dropping \'' + corr_features_df.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                to_drop_final.append(corr_features_df.iloc[:, variables].columns[maxloc])
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(corr_features_df.columns[variables])
        return to_drop_final


    # =========================== Load and process data  ====================================

    if target == 'penetration depth':
        str_index = 'dp'
    else:
        str_index = 'IBE'

    sys.stdout = open('Output_' + str_index + '.txt', 'w')

    ## Import data

    df_mp0 = pd.read_excel("Dataframe.xlsx", index_col=[0])

    ### Data cleaning

    df_mp0 = df_mp0[((df_mp0['penetration depth'] < 110) & (df_mp0['particle diameter'] == 8)) |
                    ((df_mp0['penetration depth'] < 40) & (df_mp0['particle diameter'] == 4)) |
                    ((df_mp0['penetration depth'] < 230) & (df_mp0['particle diameter'] == 16))]

    ## Define input data and output data

    y = df_mp0[target].values

    excluded = ['particle material', 'substrate material', 'penetration depth', 'particle composition',
                'substrate composition', 'Ac', 'IBE', 'E0', 'Ef', 'particle diameter']
    X = df_mp0.drop(excluded, axis=1)

    # ================= Filter_based feature selection: removing collinear features  =============================

    # construct a correlation matrix

    Xy = X.assign(**{target: y})
    corr_mat_abs_sorted = Xy.corr(method=corr_method).abs().sort_values(target, ascending=False)
    X = Xy[corr_mat_abs_sorted.drop(target, axis=0).index]

    corr_matrix = X.corr(method=corr_method)

    # plot the correlations
    fig, ax = plt.subplots()
    h = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(X, rotation=60, fontsize=14, ha='right')
    ax.set_yticks(range(len(X.columns)))
    ax.set_yticklabels(X, fontsize=14)
    plt.colorbar(h, label=f'{corr_method} correlation coefficient')
    plt.show()
    plt.savefig('fig1_' + str_index + '.png', dpi=500)

    corr_matrix_abs = corr_matrix.abs()
    upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(np.bool_))

    # to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

    upper_tri1 = upper_tri
    to_drop = []
    corr_cluster_list = []
    for column in upper_tri1.columns:
        if any(upper_tri1[column] > corr_threshold):
            to_drop.append(column)
            upper_tri1 = upper_tri1.drop(column)

            index_labels = upper_tri1[upper_tri1[column] > corr_threshold].index.tolist()
            index_labels.append(column)
            corr_cluster_list.append(index_labels)

    print("Removed features due to corr: ", to_drop)

    X_full = X

    X = X.drop(to_drop, axis=1)

    print("There are {} possible descriptors:\n\n{}".format(X.shape[1], X.columns.values))

    to_drop1 = corr_vif_features(X)
    
    # EFS_max_features = X.shape[1]

    # ================= Try some different machine learning models =============================

    ## linear regression model using scikit-learn

    lr = LinearRegression()

    # compute cross validation scores for linear regression model

    scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=n_cpu)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    r2_scores = cross_val_score(lr, X, y, scoring='r2', cv=crossvalidation, n_jobs=n_cpu)
    mae_scores = cross_val_score(lr, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=n_cpu)

    print('linear model Cross-validation results:')
    print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
    print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))
    print('Folds: %i, mean MAE: %.3f' % (len(scores), -mae_scores.mean()))

    # Plot actual vs predicted

    # plt.close('all')
    plt.figure()
    y_pred = cross_val_predict(lr, X, y, cv=crossvalidation)
    plt.scatter(y, y_pred, c='#FF2E63', alpha=0.8)
    plt.xlabel("Actual from MD simulation")
    plt.ylabel("prediction from linear model")
    bound_min = min(y.min(), y_pred.min())
    bound_max = max(y.max(), y_pred.max())
    bounds = (bound_min, bound_max)
    plt.plot(bounds, bounds, 'k-')
    plt.xlim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.ylim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.gca().axes.set_aspect('equal', 'box')
    plt.title('Linear regression')
    plt.show()
    plt.savefig('fig2_' + str_index + '.png')

    writer = pd.ExcelWriter('Parity.xlsx', engine='xlsxwriter')
    pd.DataFrame({"Observed": y, "Predicted": y_pred}).to_excel(writer, sheet_name="linear")

    ## Random forest model

    rf = RandomForestRegressor(n_estimators=100, random_state=1)

    # compute cross validation scores for random forest model

    r2_scores = cross_val_score(rf, X, y, scoring='r2', cv=crossvalidation, n_jobs=n_cpu)
    scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=n_cpu)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    mae_scores = cross_val_score(rf, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=n_cpu)

    print('Random forest Cross-validation results:')
    print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
    print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))
    print('Folds: %i, mean MAE: %.3f' % (len(scores), -mae_scores.mean()))

    # Plot actual vs predicted

    plt.figure()
    y_pred = cross_val_predict(rf, X, y, cv=crossvalidation)
    plt.scatter(y, y_pred, c='#FF2E63', alpha=0.8)
    plt.xlabel("Actual from MD simulation")
    plt.ylabel("prediction from random forest model")
    bound_min = min(y.min(), y_pred.min())
    bound_max = max(y.max(), y_pred.max())
    bounds = (bound_min, bound_max)
    plt.plot(bounds, bounds, 'k-')
    plt.xlim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.ylim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.gca().axes.set_aspect('equal', 'box')
    plt.title('Random forest regression')
    plt.show()
    plt.savefig('fig3_' + str_index + '.png')
    pd.DataFrame({"Observed": y, "Predicted": y_pred}).to_excel(writer, sheet_name="RF")

    # The most important features used by the random forest model

    rf.fit(X, y)
    importances = rf.feature_importances_
    include = X.columns.values
    indices = np.argsort(importances)[::-1]

    data = importances[indices][0:15]
    labels = include[indices][0:15]
    plt.figure()
    plt.xticks(range(len(data)), labels, rotation=30, ha='right')
    plt.ylabel('Importance (%)')
    plt.title('Feature by importances')
    plt.bar(range(len(data)), data)
    plt.show()
    plt.savefig('fig4_' + str_index + '.png')
    
    ## GPR model using scikit-learn

    kernel = 1**2 * RBF() + WhiteKernel()
    gaussian_process = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_reset_optimizer, normalize_y=False, alpha=0.0, random_state=1))

    # compute cross validation scores for GPR regression model

    scores = cross_val_score(gaussian_process, X, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=n_cpu)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    r2_scores = cross_val_score(gaussian_process, X, y, scoring='r2', cv=crossvalidation, n_jobs=n_cpu)
    mae_scores = cross_val_score(gaussian_process, X, y, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=n_cpu)

    print('GPR model Cross-validation results:')
    print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
    print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))
    print('Folds: %i, mean MAE: %.3f' % (len(scores), -mae_scores.mean()))

    # Plot actual vs predicted

    # plt.close('all')
    plt.figure()
    y_pred = cross_val_predict(gaussian_process, X, y, cv=crossvalidation)
    plt.scatter(y, y_pred, c='#FF2E63', alpha=0.8)
    plt.xlabel("Actual from MD simulation")
    plt.ylabel("prediction from GPR model")
    bound_min = min(y.min(), y_pred.min())
    bound_max = max(y.max(), y_pred.max())
    bounds = (bound_min, bound_max)
    plt.plot(bounds, bounds, 'k-')
    plt.xlim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.ylim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.gca().axes.set_aspect('equal', 'box')
    plt.title('GPR regression')
    plt.show()
    plt.savefig('fig5_' + str_index + '.png')
    pd.DataFrame({"Observed": y, "Predicted": y_pred}).to_excel(writer, sheet_name="GPR")
    
    gaussian_process.fit(X, y)
    print('GPR optimized Kernel:')
    print(gaussian_process[1].kernel_)

    # ================= wrapper_based feature selection: Exhaustive Feature Selection  =============================

    ## feature selection using mlxtend

    GPR = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_reset_optimizer, normalize_y=False, alpha=0.0, random_state=1))

    efs1 = EFS(GPR,
               min_features=1,
               max_features=EFS_max_features,
               scoring='neg_mean_squared_error',
               print_progress=True,
               cv=crossvalidation,
               n_jobs=n_cpu)

    efs1 = efs1.fit(X, y)

    print('Best accuracy score: %.2f' % efs1.best_score_)
    print('Best subset (indices):', efs1.best_idx_)
    print('Best subset (corresponding names):', efs1.best_feature_names_)

    df_FS_result = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
    df_FS_result['RSME'] = [np.mean(np.sqrt(abs(cvscores))) for cvscores in df_FS_result['cv_scores']]
    df_FS_result.sort_values('RSME', inplace=True, ascending=True)
    df_FS_result.to_excel('EFS_results_' + str_index + '.xlsx')

    metric_dict = efs1.get_metric_dict()

    ## Final Random forest model

    X_reduced = efs1.transform(X)

    GPR_f = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_reset_optimizer, normalize_y=False, alpha=0.0, random_state=1))

    # compute cross validation scores for random forest model

    r2_scores = cross_val_score(GPR_f, X_reduced, y, scoring='r2', cv=crossvalidation, n_jobs=n_cpu)
    scores = cross_val_score(GPR_f, X_reduced, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=n_cpu)
    rmse_scores = [np.sqrt(abs(s)) for s in scores]
    mae_scores = cross_val_score(GPR_f, X_reduced, y, scoring='neg_mean_absolute_error', cv=crossvalidation,
                                 n_jobs=n_cpu)

    print('Final GRP Cross-validation results:')
    print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
    print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))
    print('Folds: %i, mean MAE: %.3f' % (len(scores), -mae_scores.mean()))

    # Plot actual vs predicted

    plt.figure()
    y_pred = cross_val_predict(GPR_f, X_reduced, y, cv=crossvalidation)
    plt.scatter(y, y_pred, c='#FF2E63', alpha=0.8)
    plt.xlabel("Actual from MD simulation")
    plt.ylabel("prediction from GPR model")
    bound_min = min(y.min(), y_pred.min())
    bound_max = max(y.max(), y_pred.max())
    bounds = (bound_min, bound_max)
    plt.plot(bounds, bounds, 'k-')
    plt.xlim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.ylim((bound_min - 0.05 * (bound_max-bound_min), bound_max + 0.05 * (bound_max-bound_min)))
    plt.gca().axes.set_aspect('equal', 'box')
    plt.title('Final GPR regression')
    plt.show()
    plt.savefig('fig6_' + str_index + '.png')
    pd.DataFrame({"Observed": y, "Predicted": y_pred}).to_excel(writer, sheet_name="Final_GPR")
    
    GPR_f.fit(X, y)
    print('Final GPR optimized Kernel:')
    print(GPR_f[1].kernel_)

    writer.close()
    sys.stdout.close()
