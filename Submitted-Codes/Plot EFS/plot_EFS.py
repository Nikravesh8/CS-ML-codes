if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from ast import literal_eval

    # =========================== Inputs  ====================================

    target = 'penetration depth'  # 'penetration depth' or 'IBE'

    # =========================== Load and process data  ====================================

    ## Import data

    if target == 'penetration depth':
        str_index = 'dp'
    else:
        str_index = 'IBE'

    EFS_data = pd.read_excel('EFS_results_' + str_index + '.xlsx', index_col=[0])

    EFS_data['n_features'] = [len(literal_eval(ids)) for ids in EFS_data['feature_idx']]
    max_nfeatures = EFS_data['n_features'].max()

    # Optimum_features = pd.DataFrame()
    A = []
    B = []
    C = []
    for i in range(1, max_nfeatures+1):
        A.append(EFS_data.loc[EFS_data['n_features'] == i].iloc[0]['n_features'])
        B.append(EFS_data.loc[EFS_data['n_features'] == i].iloc[0]['feature_names'])
        C.append(EFS_data.loc[EFS_data['n_features'] == i].iloc[0]['RSME'])

    Optimum_features = pd.DataFrame({'n_features': A, 'feature_names_opt': B, 'RSME_opt': C})
    Optimum_features.to_excel('opt features_' + str_index + '.xlsx')

    plt.figure()
    plt.rcParams['font.size'] = '12'
    plt.scatter(EFS_data['n_features'], EFS_data['RSME'], c='#FF2E63', alpha=0.2)
    plt.xticks(ticks=list(range(1, max_nfeatures+1)), labels=list(range(1, max_nfeatures+1)))
    plt.plot(Optimum_features['n_features'], Optimum_features['RSME_opt'], 'k-', color='blue', linewidth=3)
    plt.xlabel("Number of features")
    plt.ylabel("CV RMSE")
    plt.savefig('opt_features_' + str_index + '.png', dpi=500)
