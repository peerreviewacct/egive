#Define NEW RGID function that doesn't do three-way interactions
def run_egive(X, y, model, metric,
             predict_method = None,
             grid_size = 20,
             h = 200, w = 200, barsize = 10, fontsize=12,
             feature_limit = None,
             pdp2_band_width = 0.10, #quantile width of areas for moderated PDPs
             #pdp3_band_width = 0.30,
             pdp_ips_trim_q = 0.9, #quantile for trimming propensity score weights
             interaction_quantiles = (0.25, 0.75),
             threeway_int_limit = 10,
             propensity_samples = 1000,
             feature_imp_njobs = 1,
             propensity_njobs = 4,
             threeway_int_njobs = 4,
             adjust_threeway = True,
             int_only = True, #whether to only break feature importances into marginal + int, or to also do lin/nonlin
             full_threeway_matrix = True,
             pdp_legend = False):

  #If y only has two outcomes, then set predict_proba to True
  if predict_method == True: #if user wants to use 'predict' method even for classification settings
    predict_proba = False
  elif len(np.unique(y))>2:
    predict_proba = False #regression model
  else:
    predict_proba = True #classification model

  if predict_proba == True:
      print('Using predict_proba method for model predictions')
  if predict_proba == False:
      print('Using predict method for model predictions')

  #Check that all features have nonzero variance
  if np.any(X.var(axis = 0) == 0):
    print(f'Features {np.where(X.var(axis=0) == 0)[0]} have zero variance. Please filter the dataset and trained model to features with non-zero variance')
    return None

  runningtime_FEATURE_IMP = 0
  runningtime_PDP = 0
  runningtime_PD2ICE = 0
  runningtime_PD2RIVER = 0
  runningtime_PD3ICE = 0
  runningtime_PD3RIVER = 0

  ###PD-AVID RUNTIME##
  timestamp_FEATURE_IMP = timeit.default_timer()

  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False
    fnames = None

  if np.array(y).dtype == 'bool':
    y = 1 * y

  if is_df == True:
    for f in range(X.shape[1]):
      if X.iloc[:,f].dtype=='bool':
        X.iloc[:,f] = X.iloc[:,f].astype('int')

    #Remove any constant features
  X = X.loc[:,np.var(X,axis=0) > 0] if is_df == True else X[:,np.var(X,axis=0) > 0]

  #Define number of features as number of columns in X
  k = X.shape[1]
  k_incl = int(k) if feature_limit is None else int(feature_limit)

  if h < barsize*k_incl*1.1:
    h = barsize*k_incl*1.1

  print('Getting ready to loop through dataset features')
    #Misc prep work
  yhat_uncentered = model.predict(X = X) if predict_proba == False else model.predict_proba(X= X)[:,1]
  yhat_mean = yhat_uncentered.mean()
  yhat = yhat_uncentered - yhat_mean

  #Get PDP matrices for main plot (linear, nonlinear, interaction effects)
  graph_dfs = []
  pdp_x_list = []
  pdp_y_list = []
  pdp_w_list = []
  pdp_not_n_list = [] #denoted 'n' because it has n data points
  pdp_y_interp_n_list = [] #denoted 'n' because it has n data points
  pdp_high_list = []
  pdp_low_list = []
  pdp_ice_list = []
  h2_list = []

  pdp_master_array = np.zeros((X.shape[0], k, 2))

  #Get PDP's for each feature
  delayed_funcs = []
  from joblib import Parallel, delayed
  for f in range(k):
    delayed_funcs.append(
        # delayed(feature_importance_scores)(X = X, y = y, model = model, f = f, metric = metric,
        # grid_size = grid_size, predict_proba = predict_proba,
        # interaction_quantiles = interaction_quantiles)
        delayed(feature_importance_scores)(X = X, y = y, model = model, f = f, metric = metric,
        grid_size = grid_size, predict_proba = predict_proba,
        interaction_quantiles = interaction_quantiles, int_only = int_only)
    )
  if feature_imp_njobs == 1:
    print('Calculating for each feature in order')
  else:
    print('Calculating for features in parallel with '+str(feature_imp_njobs)+ ' CPUs')
  result_tuples = Parallel(n_jobs = feature_imp_njobs)(delayed_funcs)
  print('Computed '+str(len(result_tuples))+ ' PDPs')

  for f in range(k):
    if int_only == False:
      s_linear, s_nonlinear, s_int, s_none, pdp_x, pdp_y, pdp_not_n, pdp_y_interp_n, pdp_ice, h2, weights = result_tuples[f]
    else:
      s_marginal, s_int, s_none, pdp_x, pdp_y, pdp_not_n, pdp_y_interp_n, pdp_ice, h2, weights = result_tuples[f]

    pdp_x_list.append(pdp_x)
    pdp_y_list.append(pdp_y)
    pdp_w_list.append(weights)
    pdp_not_n_list.append(pdp_not_n)
    pdp_y_interp_n_list.append(pdp_y_interp_n)
    pdp_ice_list.append(pdp_ice)
    h2_list.append(h2)

    pdp_master_array[:, f, 0] = pdp_y_interp_n
    pdp_master_array[:, f, 1] = pdp_not_n

    if int_only == False:
      graph_dfs.append( pd.DataFrame({'Feature_Num': str(f), 'Type': ['(a) Linear','(b) Nonlin.','(c) Int.','None'],
                                      'Diff': [s_linear, s_nonlinear, s_int, s_none]}) )
    else:
      graph_dfs.append( pd.DataFrame({'Feature_Num': str(f), 'Type': ['(a) Marginal','(b) Int.','None'],
                                       'Diff': [s_marginal, s_int, s_none]}) )


  #Create PDP df for main graph
  graph_df_full = pd.concat(objs = graph_dfs, axis = 0)
  if fnames is None:
    graph_df_full['Feature'] = ['X'+str(n) for n in graph_df_full['Feature_Num'].astype('int').values]
  else:
    graph_df_full['Feature'] = [fnames[n] for n in graph_df_full['Feature_Num'].astype('int').values]

  #Find top n features from graph_df (will use to limit interaction loop)
  graph_df = graph_df_full[graph_df_full['Type']!='None']
  ranked = graph_df.groupby(['Feature','Feature_Num'],as_index = False)['Diff'].apply(lambda x: np.clip(x, a_min = 0, a_max = None).sum())
  ranked['rank'] = ranked['Diff'].rank(ascending = False)
  if feature_limit is not None:
    kept_features = ranked.loc[ranked['rank']<=feature_limit]['Feature'].values
    kept_feature_nums = ranked.loc[ranked['rank']<=feature_limit]['Feature_Num'].values
    kept_feature_nums = [int(k) for k in kept_feature_nums]
    ranked = ranked[ranked['Feature'].isin(kept_features)]
  else:
    kept_features = X.columns if is_df==True else ['X'+str(i) for i in range(k)]
    kept_feature_nums = np.arange(k)
    #ranked now only has kept features in it

  runningtime_FEATURE_IMP += timeit.default_timer() - timestamp_FEATURE_IMP

  ###PDP RUNTIME###
  timestamp_PDP = timeit.default_timer()

  #Create df for pdp plot
  ns = [len(p) for p in pdp_x_list]
  if fnames is None:
    features = np.concatenate([['X'+str(i)]*n for i,n in zip(range(k), ns)])
  else:
    features = np.concatenate([[fnames[i]]*n for i,n in zip(range(k), ns)])

  feature_num = np.concatenate([[i]*n for i,n in zip(range(k), ns)])
  pdp_df = pd.DataFrame({'Feature': features, 'Feature_Num': feature_num, \
                         'X': np.concatenate(pdp_x_list), 'Y': np.concatenate(pdp_y_list)})

  #Duplicate PDP df for 'linear' and 'nonlinear', since the same PDP will be shown for both hover actions
  pdp_df['Type'] = '(a) Linear' ; pdp_df2 = pdp_df.copy(deep = True) ; pdp_df2['Type'] = '(b) Nonlin.'
  pdp_df_combined = pd.concat(objs = [pdp_df,pdp_df2], axis = 0)

  #Find top n features from pdp plot
  if feature_limit is not None:
    pdp_df_combined = pdp_df_combined[pdp_df_combined['Feature'].isin(kept_features)]

  runningtime_PDP += timeit.default_timer() - timestamp_PDP

  ###PD2ICE and PD3ICE RUNTIME###
  timestamp_PD2ICE_PD3ICE = timeit.default_timer()

  print('Getting propensity weights')
    #Get filters and weights for estimated pairwise h^2 statistics
  h2_filter2_list = []
  h2_weight2_list = []
  delayed_funcs = []

  for f in range(k):
    delayed_funcs.append(
        delayed(propensity_filters_and_weights)(f = f, pdp_w_list = pdp_w_list,
            pdp2_band_width = pdp2_band_width, run_threeway = False,
            X = X, pdp_ips_trim_q = pdp_ips_trim_q, propensity_samples = propensity_samples) #X_residuals = f_resids,
    )
  if propensity_njobs == 1:
    print('Getting propensity weights in order')
  else:
    print('Getting propensity weights in parallel with '+str(propensity_njobs)+ ' CPUs')
  result_tuples = Parallel(n_jobs = propensity_njobs)(delayed_funcs)
  print('Computed '+str(len(result_tuples))+ ' PDPs')

  for f in range(k):
    f_filters2, f_weights2 = result_tuples[f]
    h2_filter2_list.append(f_filters2); h2_weight2_list.append(f_weights2)

  added_time = timeit.default_timer() - timestamp_PD2ICE_PD3ICE
  runningtime_PD2ICE += added_time / 2
  runningtime_PD3ICE += added_time / 2

  ###PD2RIVER RUNTIME###
  timestamp_PD2RIVER = timeit.default_timer()
    #Get filters and weights for moderated PDPs
  upper_cutoffs = np.quantile(X, axis = 0, q = interaction_quantiles[1])
  lower_cutoffs = np.quantile(X, axis = 0, q = interaction_quantiles[0])
  maxs = np.max(X, axis = 0)
  mins = np.min(X, axis = 0)

  #If they are equal, and at max, then u is unchanged and l adjusts downward
  #If they are equal, and at min, then u is changed and l is unchanged
  upper_cutoffs = np.array([u + 0.001 if (u == l) and (u != m) else u for u, l, m in zip(upper_cutoffs , lower_cutoffs, maxs)])
  lower_cutoffs = np.array([l - 0.001  if (l == u) and (l != m) else l for u, l, m in zip(upper_cutoffs , lower_cutoffs, mins)])

  upper_cutoffs = upper_cutoffs.reshape(1,-1)
  lower_cutoffs = lower_cutoffs.reshape(1,-1)

  binary_adjust_counter = 0
  for f in range(np.array(X).shape[1]):
    if len(np.unique(np.array(X)[:,f]))==2:
      binary_adjust_counter+=1
      upper_cutoffs[0,f] = np.max(X.iloc[:,f]) if is_df == True else np.max(X[:,f])
      lower_cutoffs[0,f] = np.min(X.iloc[:,f]) if is_df == True else np.min(X[:,f])
  print(f'Adjusted upper/lower cutoffs for {binary_adjust_counter} binary features')

  pdp_high_filter = np.array(X) >= upper_cutoffs
  pdp_low_filter = np.array(X) <= lower_cutoffs

  pdp_high_weights = np.zeros(pdp_high_filter.shape)
  pdp_low_weights = np.zeros(pdp_low_filter.shape)

  pdp_high_mat = np.zeros(pdp_high_filter.shape)
  pdp_low_mat = np.zeros(pdp_low_filter.shape)

  print('Getting 2PDQUIVER propensity weights')
  for f in range(k):

    high_weights, low_weights = pdp_propensity_filters_and_weights(f, pdp_high_filter, pdp_low_filter,
                                                                   X, pdp_ips_trim_q, propensity_samples)

    pdp_high_weights[:,f] = high_weights
    pdp_low_weights[:,f] = low_weights

    pdp_high_mat[:,f] = 1 * pdp_high_filter[:,f] * pdp_high_weights[:,f] #change the fth column to be the weight (for included points) or zero (for not included points)
    pdp_low_mat[:,f] = 1 * pdp_low_filter[:,f] * pdp_low_weights[:,f] #change the fth column to be the weight (for included points) or zero (for not included points)

  pdp_high_mat_w = pdp_high_mat / np.sum(pdp_high_mat, axis = 0).reshape(1,-1) #normalize the weight matrix
  pdp_low_mat_w = pdp_low_mat / np.sum(pdp_low_mat, axis = 0).reshape(1,-1) #normalize the weight matrix

  for f in range(k):
    f_pdp_high = pdp_ice_list[f] @ pdp_high_mat_w
    f_pdp_low = pdp_ice_list[f] @ pdp_low_mat_w
    pdp_high_list.append(f_pdp_high)
    pdp_low_list.append(f_pdp_low)

    #Get dataframe of moderated PDPs
  int_score_diffs = []
  f1 = []
  f2 = []
  tuples = []
  if metric=='rmse':
    metric = rmse #callable function
  if metric=='mse':
    metric = mse

  error_matrices = {}
  counter = -1

  if fnames is None:
    f1 = ['X'+str(i) for i in range(k) for j in range(k)]
    f2 = ['X'+str(j) for i in range(k) for j in range(k)]
  else:
    f1 = [fnames[i] for i in range(k) for j in range(k)]
    f2 = [fnames[j] for i in range(k) for j in range(k)]

  X_col = np.concatenate([pdp_x_list[i] for i in range(k) for j in range(k)])
  ns = [len(pdp_x_list[i]) for i in range(k) for j in range(k)]
  int_pdp_col_high = np.concatenate([pdp_high_list[i][:,j] for i in range(k) for j in range(k)])
  int_pdp_col_low = np.concatenate([pdp_low_list[i][:,j] for i in range(k) for j in range(k)])

  pdp_df_high = pd.DataFrame({
      'Feature': np.repeat(f1, ns),
      'Feature 2': np.repeat(f2, ns),
      'Feature Combo': [_f1+':'+_f2 for _f1, _f2 in zip(np.repeat(f1, ns), np.repeat(f2, ns))],
      'X': X_col,
      'Feature 2 Level': ['High']*len(X_col),
      'Y': int_pdp_col_high
  })
  pdp_df_low = pd.DataFrame({
      'Feature': np.repeat(f1, ns),
      'Feature 2': np.repeat(f2, ns),
      'Feature Combo': [_f1+':'+_f2 for _f1, _f2 in zip(np.repeat(f1, ns), np.repeat(f2, ns))],
      'X': X_col,
      'Feature 2 Level': ['Low']*len(X_col),
      'Y': int_pdp_col_low
  })

  pdp_df_twoway = pd.concat([pdp_df_high,pdp_df_low], axis = 0, ignore_index = True)
  pdp_df_twoway = pdp_df_twoway[pdp_df_twoway['Feature'] != pdp_df_twoway['Feature 2']]

  #Limit pdp df to kept features
  if feature_limit is not None:
    pdp_df_twoway = pdp_df_twoway.loc[(pdp_df_twoway['Feature'].isin(kept_features)) & (pdp_df_twoway['Feature 2'].isin(kept_features)),:]
  print('PDP df shape: ' + str(pdp_df_twoway.shape))

  runningtime_PD2RIVER += timeit.default_timer() - timestamp_PD2RIVER

  ###PD2ICE RUNTIME###
  timestamp_PD2ICE = timeit.default_timer()

  print('Running through interactions')
  f1_short = [] ; f2_short = [] #these will only store the interactions we loop through (k*(k-1)/2), not all k**2
  if feature_limit is None:
    kept_feature_nums = np.arange(k)
  feature_int_error_cors = {}
  for i in kept_feature_nums:
    for j in range(i):
      if j in kept_feature_nums:
        counter+=1 #starts at counter = 0
        if counter%500==0:
          print('Interaction '+str(counter))
        if fnames is None:
          f1_short.append('X'+str(i))
          f2_short.append('X'+str(j))
        else:
          f1_short.append(fnames[i])
          f2_short.append(fnames[j])

        name_tuple = np.sort([f1_short[-1], f2_short[-1]])
        f_tuple = (i,j) ; tuples.append(name_tuple[0]+":"+name_tuple[1])
        f1_ice = pdp_ice_list[i]
        f2_ice = pdp_ice_list[j]

        ###################################
        f1_filters, f1_weights = h2_filter2_list[i], h2_weight2_list[i]
        f2_filters, f2_weights = h2_filter2_list[j], h2_weight2_list[j]

        pdp2_f1_f2 = np.array([np.average(f1_ice[:,f], axis = 1, weights = w[f]) for f,w in zip(f2_filters, f2_weights)]).T #f1 increases down, f2 increases across
        pdp2_f2_f1 = np.array([np.average(f2_ice[:,f], axis = 1, weights = w[f]) for f,w in zip(f1_filters, f1_weights)]) #f2 increases across, f1 increases down
        pdp2_final = (pdp2_f1_f2 + pdp2_f2_f1) / 2

          #Calculate H^2 score
        pdp2_final = pdp2_final - pdp2_final.mean()
        pdp2_f1 = (pdp2_final.mean(axis = 1) - pdp2_final.mean(axis = 1).mean()).reshape(-1,1)
        pdp2_f2 = (pdp2_final.mean(axis = 0) - pdp2_final.mean(axis = 0).mean()).reshape(1,-1)

        resid_h2 = np.sum((pdp2_final - pdp2_f1 - pdp2_f2)**2) / np.sum(pdp2_final**2)

        #################

        #Interaction-length list, where each element is a feature-length list
        int_score_diffs.append(resid_h2)


  int_df = pd.DataFrame({'Feature': f1_short, 'Feature 2': f2_short, 'Feature Combo': tuples, 'Diff': int_score_diffs})
  #At this point, int_df is missing the 'reversed' copies of itself. That is easy to fix - just flip Feature and Feature 2.
  int_df_copied = int_df.copy(deep = True)
  featcol = int_df['Feature'].values ; feat2col = int_df['Feature 2'].values
  int_df_copied.loc[:,'Feature'] = feat2col ; int_df_copied.loc[:,'Feature 2'] = featcol
  int_df = pd.concat([int_df, int_df_copied], axis = 0)
  int_df.loc[int_df['Feature']==int_df['Feature 2'], 'Diff'] = 0 #zero out same-variable rows; variable can't interact with itself
  orig_int_df = int_df.copy(deep = True)

  runningtime_PD2ICE += timeit.default_timer() - timestamp_PD2ICE

  ###PD3ICE RUNTIME###
  timestamp_PD3ICE = timeit.default_timer()

  #Higher-order interaction plot
  int_df_filt = int_df[int_df['Feature'] != int_df['Feature 2']]
  int_df_filt = int_df_filt[int_df_filt['Diff'] > 0 ]
  int_df_filt = int_df_filt.drop_duplicates('Feature Combo')
  int_df_filt = int_df_filt.sort_values('Diff', ascending = False)
  best_rmse = []
  best_rmse_unadj = []
  best_higherorder_interaction = []
  best_pdp_low = []
  best_pdp_high = []


  threeway_int_score_matrix = np.zeros((min([int_df_filt.shape[0], threeway_int_limit]), len(kept_feature_nums)))
  # avg_cor_matrix = np.zeros((min([int_df_filt.shape[0], threeway_int_limit]), len(kept_feature_nums)))
  # adj_factor_matrix = np.zeros((min([int_df_filt.shape[0], threeway_int_limit]), len(kept_feature_nums)))

  int_tuple_name_list = []
  int_tuple_list = []

  #delayed_funcs = []
  for ix in range(min([int_df_filt.shape[0], threeway_int_limit])):
    int_tuple_name = ( int_df_filt.iloc[ix,0], int_df_filt.iloc[ix,1] )
    if fnames is None:
      int_tuple = ( int(int_tuple_name[0].replace('X','')), int(int_tuple_name[1].replace('X','')) )
    else:
      int_tuple = (
          int(np.where(np.array(fnames)==int_tuple_name[0])[0]),
          int(np.where(np.array(fnames)==int_tuple_name[1])[0]),
      )
    int_tuple_name_list.append(int_tuple_name)
    int_tuple_list.append(int_tuple)

  #Note: higher_order_df now just lists the candidate interactions, WITHOUT any h^2 scores
  higher_order_df = pd.DataFrame({'Feature': list(kept_features) * threeway_int_score_matrix.shape[0],
                                     'Feature Num': list(kept_feature_nums) * threeway_int_score_matrix.shape[0],
                                     'Score': threeway_int_score_matrix.reshape(-1),
                                      'Interaction': np.repeat([str(_int) for _int in int_tuple_name_list], len(kept_features)),
                                     'Interaction V1': np.repeat([_int[0] for _int in int_tuple_list], len(kept_features)),
                                     'Interaction V2': np.repeat([_int[1] for _int in int_tuple_list], len(kept_features)),
                                     'Feature Combo': list(zip(
                                        list(kept_feature_nums) * threeway_int_score_matrix.shape[0],
                                        np.repeat([_int[0] for _int in int_tuple_list], len(kept_features)),
                                        np.repeat([_int[1] for _int in int_tuple_list], len(kept_features))
                                     ))})
  top_df = higher_order_df.copy(deep =True) #used to filter this interaction DF, but not anymore
  top_threeway_ints = top_df['Feature Combo']

  #Swap features 0 and 1
    #Re-do code below when using column names
  top_df2 = top_df.copy(deep = True)
  top_df2['Feature Num'] = [t[1] for t in top_threeway_ints] #put second feature as main feature
  top_df2['Feature'] = ['X'+ str(n) for n in list(top_df2['Feature Num'])] if fnames is None else [fnames[n] for n in list(top_df2['Feature Num'])]
  top_df2['Interaction V1'] = top_df['Feature Num'] #put first feature as second feature
  if fnames is None:
    top_df2['Interaction'] = [str(('X'+str(i1), 'X'+str(i2))) for i1, i2 in zip(top_df2['Interaction V1'], top_df2['Interaction V2'])]
  else:
    top_df2['Interaction'] = [str((fnames[i1], fnames[i2])) for i1, i2 in zip(top_df2['Interaction V1'], top_df2['Interaction V2'])]

  #Swap features 0 and 2
     #Re-do code below when using column names
  top_df3 = top_df.copy(deep = True)
  top_df3['Feature Num'] = [t[2] for t in top_threeway_ints] #put third feature as main feature
  top_df3['Feature'] = ['X'+ str(n) for n in list(top_df3['Feature Num'])] if fnames is None else [fnames[n] for n in list(top_df3['Feature Num'])]
  top_df3['Interaction V2'] = top_df['Feature Num'] #put first feature as third feature
  if fnames is None:
    top_df3['Interaction'] = [str(('X'+str(i1), 'X'+str(i2))) for i1, i2 in zip(top_df3['Interaction V1'], top_df3['Interaction V2'])]
  else:
    top_df3['Interaction'] = [str((fnames[i1], fnames[i2])) for i1, i2 in zip(top_df3['Interaction V1'], top_df3['Interaction V2'])]

  higher_order_df = pd.concat([higher_order_df, top_df2, top_df3]).drop_duplicates(['Feature','Interaction V1','Interaction V2'])

  runningtime_PD3ICE += timeit.default_timer() - timestamp_PD3ICE

  ###PD3RIVER RUNTIME###
  timestamp_PD3RIVER = timeit.default_timer()

  #Create matrix of moderated PDPs
  print('Filtering three-way interactions for PDP matrix')
  high_int_dfs = []
  #higher_order_df_filtered_for_pdps = higher_order_df[higher_order_df['Score'] > 0]
  higher_order_df_filtered_for_pdps = higher_order_df.copy(deep = True) #[higher_order_df['Score'] > 0]

  # max_int_n = int(np.floor( 5000 / (grid_size*4) ))
  # if higher_order_df_filtered_for_pdps.shape[0] > max_int_n:
  #   higher_order_df_filtered_for_pdps = higher_order_df_filtered_for_pdps.sort_values('Score', ascending = False).iloc[0:max_int_n,:]

  print('Calculated three-way moderated PDPs')
  for i in range(higher_order_df_filtered_for_pdps.shape[0]):
    f_name = higher_order_df_filtered_for_pdps['Feature'].iloc[i]
    f = higher_order_df_filtered_for_pdps["Feature Num"].iloc[i]
    i1 = higher_order_df_filtered_for_pdps["Interaction V1"].iloc[i]
    i2 = higher_order_df_filtered_for_pdps["Interaction V2"].iloc[i]

    int_filter_high_high = pdp_high_filter[:,i1] & pdp_high_filter[:,i2]
    int_filter_low_high = pdp_low_filter[:,i1] & pdp_high_filter[:,i2]
    int_filter_high_low = pdp_high_filter[:,i1] & pdp_low_filter[:,i2]
    int_filter_low_low = pdp_low_filter[:,i1] & pdp_low_filter[:,i2]

    #Only compute two-way PDP if all four high/low combinations are available
    ice_matrix = pdp_ice_list[f]
    x = pdp_df_combined[pdp_df_combined['Feature']==f_name]['X'].unique()

    high_int_df_combined = []
    if (int_filter_high_high.sum() > 0):
      int_weights_high_high = (pdp_high_weights[:, i1] * pdp_high_weights[:,i2])[int_filter_high_high]
      int_pdp_high_high = np.average(ice_matrix[:,int_filter_high_high], axis = 1, weights = int_weights_high_high)
      #int_pdp_high_high = int_pdp_high_high - int_pdp_high_high.mean()
      high_int_df_HIGH_HIGH = pd.DataFrame({'Feature': f_name, 'Feature Num': f, 'V2 Level': 'High', 'V3 Level': 'High', 'X': x, 'Y': int_pdp_high_high, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_HIGH_HIGH)

    if (int_filter_low_high.sum() > 0):
      int_weights_low_high = (pdp_low_weights[:, i1] * pdp_high_weights[:, i2])[int_filter_low_high]
      int_pdp_low_high = np.average(ice_matrix[:,int_filter_low_high], axis = 1, weights = int_weights_low_high)
      #int_pdp_low_high = int_pdp_low_high - int_pdp_low_high.mean()
      high_int_df_LOW_HIGH = pd.DataFrame({'Feature': f_name, 'Feature Num': f, 'V2 Level': 'Low', 'V3 Level': 'High', 'X': x, 'Y': int_pdp_low_high, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_LOW_HIGH)

    if (int_filter_high_low.sum() > 0):
      int_weights_high_low = (pdp_high_weights[:, i1] * pdp_low_weights[:, i2])[int_filter_high_low]
      int_pdp_high_low = np.average(ice_matrix[:,int_filter_high_low], axis = 1, weights = int_weights_high_low)
      #int_pdp_high_low = int_pdp_high_low - int_pdp_high_low.mean()
      high_int_df_HIGH_LOW = pd.DataFrame({'Feature': f_name, 'Feature Num': f, 'V2 Level': 'High', 'V3 Level': 'Low', 'X': x, 'Y': int_pdp_high_low, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_HIGH_LOW)

    if (int_filter_low_low.sum() > 0):
      int_weights_low_low = (pdp_low_weights[:, i1] * pdp_low_weights[:, i2])[int_filter_low_low]
      int_pdp_low_low = np.average(ice_matrix[:,int_filter_low_low], axis = 1, weights = int_weights_low_low)
      #int_pdp_low_low = int_pdp_low_low - int_pdp_low_low.mean()
      high_int_df_LOW_LOW = pd.DataFrame({'Feature': f_name, 'Feature Num': f, 'V2 Level': 'Low', 'V3 Level': 'Low', 'X': x, 'Y': int_pdp_low_low, 'Interaction V1': i1, 'Interaction V2': i2})
      high_int_df_combined.append(high_int_df_LOW_LOW)

    if len(high_int_df_combined)>0:
      high_int_dfs.append(pd.concat(high_int_df_combined, axis = 0))

  if len(high_int_dfs) > 0:
    high_int_df_long = pd.concat(high_int_dfs, axis = 0)
    print(f'Three-way PDP df has {high_int_df_long.shape[0]} rows')
  else:
    print('No three-way PDPs to visualize')
    high_int_df_long = pd.DataFrame(columns = ['Feature','Feature Num','V2 Level','V3 Level','X','Y','Interaction V1','Interaction V2'])

  runningtime_PD3RIVER += timeit.default_timer() - timestamp_PD3RIVER

  ####PLOT RESULTS###
  #Disable altair max rows
  alt.data_transformers.disable_max_rows()

  ###PD-AVID RUNTIME - PLOTTING###
  timestamp_FEATURE_IMP = timeit.default_timer()

  #Plot results
  print('Generating plot')
  feature_selector1 = alt.selection_single(on="mouseover", encodings=['y'])
  feature_selector1_click = alt.selection_multi(on="click", encodings=['y'])
  feature_selector2 = alt.selection_single(on="mouseover", encodings = ['color'])
  feature_selector2_click = alt.selection_multi(on="click", encodings = ['color'])
  feature_selector3 = alt.selection_single(on="mouseover", encodings=['y'])
  feature_selector3_click = alt.selection_multi(on="click", encodings=['y'])
  if full_threeway_matrix==True: #plot 5 will be a matrix-style heatmap, and need to filter variables 2 & 3 based on x and y
    feature_selector4 = alt.selection_single(on="mouseover", encodings=['x','y'])
    feature_selector4_click = alt.selection_multi(on="click", encodings=['x','y'])
  if full_threeway_matrix==False: #plot 5 will be a bar chart, and need to filter just based on the y-axis
    feature_selector4 = alt.selection_single(on="mouseover", encodings=['y'])
    feature_selector4_click = alt.selection_multi(on="click", encodings=['y'])

  #global _plot_feature_selectors
  _plot_feature_selectors = [feature_selector1, feature_selector1_click, feature_selector2,
                              feature_selector2_click, feature_selector3, feature_selector3_click, feature_selector4, feature_selector4_click]

  if int_only == False:
    dom = ['(a) Linear','(b) Nonlin.','(c) Int.']
    ran = ['#1f77b4FF', '#17becfFF', '#2ca02cFF']
  else:
    dom = ['(a) Marginal','(b) Int.']
    ran = ['#1f77b4FF', '#2ca02cFF']

  #Bar charts
  sort_order = ranked.sort_values('rank', ascending = True)['Feature'].values
    #limited to top n if feature_limit is not none
  if feature_limit is not None:
    graph_df = graph_df[graph_df['Feature'].isin(kept_features)]

    #This is the left-most feature importance decomposition chart
  decomp_xmin = graph_df.groupby('Feature')['Diff'].apply(lambda x: x[x<0].sum()).min()
  decomp_xmax = graph_df.groupby('Feature')['Diff'].apply(lambda x: x[x>0].sum()).max()

  _opac = alt.condition(feature_selector1_click, alt.value(1.0), alt.value(0.5))
  decomp_chart = alt.Chart(graph_df, title = 'Plot 1: Feature Importances').mark_bar(size=barsize).encode(
      x=alt.X('sum(Diff)', scale = alt.Scale(domain = (decomp_xmin, decomp_xmax)), axis = alt.Axis(title = 'Score')),
      y=alt.Y('Feature',sort = sort_order),
      color=alt.Color('Type', scale = alt.Scale(domain = dom, range = ran),
                      legend=alt.Legend(title = None, orient='top', labelFontSize = fontsize, titleFontSize = fontsize)),
      opacity = _opac
      ).properties(
      width=w,
      height=h
  )
  decomp_chart = decomp_chart.add_selection(feature_selector1_click)

  runningtime_FEATURE_IMP += timeit.default_timer() - timestamp_FEATURE_IMP
  timestamp_PDP = timeit.default_timer()

  ###PDP RUNTIME - PLOTTING###
  pdp_xmin = pdp_df_combined['Y'].min()
  pdp_xmax = pdp_df_combined['Y'].max()

  pdp_legend_object = None if pdp_legend == False else alt.Legend(title = None, orient = 'right', labelFontSize = fontsize, titleFontSize = fontsize)

  pdp_chart = alt.Chart(pdp_df_combined, title = 'Plot 2: Partial Dep. Plots (PDPs)').mark_line().encode(
      alt.X('X'),
      alt.Y('Y', scale = alt.Scale(zero = False, domain = (pdp_xmin, pdp_xmax))),
      color=alt.condition(feature_selector2|feature_selector2_click,alt.Color('Feature:N',
          #legend = None
          legend = pdp_legend_object,
          ),alt.value('lightgray'))).properties(width=w,height=h)

  pdp_chart = pdp_chart.transform_filter(feature_selector1_click)

  pdp_chart = pdp_chart.add_selection(
              feature_selector2,feature_selector2_click)

  runningtime_PDP += timeit.default_timer() - timestamp_PDP

  ###PD2ICE RUNTIME - PLOTTING###
  timestamp_PD2ICE = timeit.default_timer()

  scale = alt.Scale(domain = int_df['Feature'].unique(),
                                    range = list(['#2ca02cFF']*k))
  int_df_positive = int_df[int_df['Diff'] > 0]

  #if int_df_positive.shape[0] > 5000:
  #  int_df_positive = int_df_positive.sort_values('Diff', ascending = False).iloc[0:5000,:]
  #  print('Filtering to '+str(int_df_positive.shape[0])+ ' interaction scores')

  _col = alt.condition(feature_selector3_click,
                          alt.Color('Feature:N',scale = scale,legend = None),
                          alt.value('lightgray'))
  int_chart = alt.Chart(data = int_df_positive, title = 'Plot 3: Pairwise Int. Scores').mark_bar(size = barsize).encode(
      x=alt.X('sum(Diff)',
      axis = alt.Axis(title = 'Score')),
      y=alt.Y('Feature 2', title = 'Interaction Feature', sort = sort_order),
      color=_col
      ).properties(width=w,height=h)

  int_chart = int_chart.transform_filter(feature_selector1_click)
  int_chart = int_chart.add_selection(feature_selector3, feature_selector3_click)

  runningtime_PD2ICE += timeit.default_timer() - timestamp_PD2ICE

  ###PD2RIVER RUNTIME - PLOTTING###
  timestamp_PD2RIVER = timeit.default_timer()
  int_names = list(error_matrices.keys())
  positive_ints = list(int_df[int_df['Diff']>0]['Feature Combo'])
  positive_ints = positive_ints + [p.split(':')[1]+':'+p.split(':')[0] for p in positive_ints]

  pdp_df_twoway_incl = pdp_df_twoway[pdp_df_twoway['Feature Combo'].isin(positive_ints)]
  print('Filtered PDP df has '+str(pdp_df_twoway_incl.shape[0])+ ' rows')
  print('based on '+ str(pdp_df_twoway_incl['Feature Combo'].nunique())+ ' interactions with positive score')

  max_int_n = int(np.floor( 5000 / (grid_size*2*2) )) #each interaction requires 2*grid_size rows, and each interaction can appear twice

  # if pdp_df_twoway_incl.shape[0] > 5000:
  #     #Get top interactions (note: each interaction is counted twice, because it can be visualized two ways)
  #   top_n_ints = list(int_df.sort_values('Diff', ascending = False)['Feature Combo'].iloc[0:max_int_n])
  #     #This includes both 'copies' of the interaction, since pdp_df Feature Combos are ordered
  #   top_n_ints = top_n_ints + [p.split(':')[1]+ ':' + p.split(':')[0] for p in top_n_ints]
  #   pdp_df_twoway_incl = pdp_df_twoway_incl[pdp_df_twoway_incl['Feature Combo'].isin(top_n_ints)]

  #   done = False
  #   counter = -1
  #   while done == False:
  #     counter +=1
  #     next_index = max_int_n + counter
  #     next_int = int_df.sort_values('Diff', ascending = False)['Feature Combo'].iloc[next_index]
  #     next_int = [next_int] + [next_int.split(':')[1]+ ':' + next_int.split(':')[0]]
  #     pdp_df_added = pdp_df_twoway[pdp_df_twoway['Feature Combo'].isin(next_int)]

  #     if pdp_df_added.shape[0] + pdp_df_twoway_incl.shape[0] <=5000:
  #       pdp_df_twoway_incl = pd.concat([pdp_df_twoway_incl, pdp_df_added], axis = 0)
  #     else:
  #       done = True

  #   print('Filtered to '+str(pdp_df_twoway_incl['Feature Combo'].nunique()) + ' interactions')
  #   print('PDP df now has '+str(pdp_df_twoway_incl.shape[0]) + ' rows')

  ymin = pdp_df_twoway_incl['Y'].min()
  ymax = pdp_df_twoway_incl['Y'].max()

  color_dom = list(pdp_df_twoway_incl['Feature'].unique())
  color_ran = ['#1f77b4BF'] * len(color_dom)

  pdp_twoway_plot = alt.Chart(pdp_df_twoway_incl, title = 'Plot 4: Pairwise PDPs').mark_line().encode(
  x = alt.X('X'),
  y = alt.Y('Y', title = 'Y', scale=alt.Scale(domain=(ymin,ymax))),
  color = alt.Color('Feature:N', scale = alt.Scale(domain = color_dom, range = color_ran), legend = None),
  detail = alt.Detail('Feature 2'),
  strokeDash = alt.StrokeDash('Feature 2 Level',
       legend = alt.Legend(orient= 'none', legendX = w + 10, legendY = 0.05 * h, title = 'V2 Level',
                           symbolStrokeColor = 'black', symbolSize = 400)
    ) #, labelFontSize = fontsize, titleFontSize = fontsize)
  ).properties(width=w, height=h )
  pdp_twoway_plot = pdp_twoway_plot.transform_filter(feature_selector1_click).transform_filter(feature_selector3_click)

  runningtime_PD2RIVER += timeit.default_timer() - timestamp_PD2RIVER

  ###PD3ICE RUNTIME - PLOTTING###
  timestamp_PD3ICE = timeit.default_timer()

  #h2_min, h2_max = higher_order_df.Score.min(), higher_order_df.Score.quantile(0.99)

  #Get feature names
  high_int_df_long['Feature 2'] = ['X'+str(i) for i in list(high_int_df_long['Interaction V1'])] if fnames is None else [fnames[i] for i in list(high_int_df_long['Interaction V1'])]
  high_int_df_long['Feature 3'] = ['X'+str(i) for i in list(high_int_df_long['Interaction V2'])] if fnames is None else [fnames[i] for i in list(high_int_df_long['Interaction V2'])]
  high_int_df_long = high_int_df_long[ (high_int_df_long['Feature Num'] != high_int_df_long['Interaction V1'] ) & (high_int_df_long['Feature Num'] != high_int_df_long['Interaction V2'])]
  sort_order_v2 = ['X'+str(i) for i in np.sort(high_int_df_long['Interaction V1'].unique())] if fnames is None else [fnames[i] for i in np.sort(high_int_df_long['Interaction V1'].unique())]
  sort_order_v3 = ['X'+str(i) for i in np.sort(high_int_df_long['Interaction V2'].unique())] if fnames is None else [fnames[i] for i in np.sort(high_int_df_long['Interaction V2'].unique())]

  #Create three-way score plot
  scale = alt.Scale(domain = high_int_df_long['Feature'].unique(),
                                    range = list(['#2ca02cFF']*k))
  _col = alt.condition(feature_selector4_click,
                          alt.Color('Feature:N',scale = scale,legend = None),
                          alt.value('lightgray'))

  #Append three-way scores, and group PDP dataframe by interaction
  high_int_df_long_3scores = egive_append_threeway_scores(high_int_df_long)
  high_int_df_long_3scores = high_int_df_long_3scores.groupby(['Feature','Feature 2','Feature 3'], as_index = False)['Threeway Score'].mean()

  high_int_chart = alt.Chart(data = high_int_df_long_3scores, title = 'Plot 5: Three-way Int. Scores').mark_bar(size = barsize).encode(
      x=alt.X('sum(Threeway Score)',
      axis = alt.Axis(title = 'Score')),
      y=alt.Y('Feature 3', title = '3rd Interaction Feature', sort = sort_order),
      color=_col
      ).properties(width=w,height=h)

  high_int_chart = high_int_chart.transform_filter(feature_selector1_click)
  high_int_chart = high_int_chart.transform_filter(feature_selector3_click)
  high_int_chart = high_int_chart.add_selection(feature_selector4, feature_selector4_click)

  runningtime_PD3ICE += timeit.default_timer() - timestamp_PD3ICE


  ###PD3RIVER RUNTIME - PLOTTING###
  timestamp_PD3RIVER = timeit.default_timer()

  #Create chart showing stratified PDPs by the best two-way interaction for each feature
  ymin = high_int_df_long['Y'].min()
  ymax = high_int_df_long['Y'].max()

  #Get feature names
  # if fnames is None:
  #   high_int_df_long['Variable 2'] = ['X'+str(i) for i in list(high_int_df_long['Interaction V1'])]
  #   high_int_df_long['Variable 3'] = ['X'+str(i) for i in list(high_int_df_long['Interaction V2'])]
  # else:
  #   high_int_df_long['Variable 2'] = [fnames[i] for i in list(high_int_df_long['Interaction V1'])]
  #   high_int_df_long['Variable 3'] = [fnames[i] for i in list(high_int_df_long['Interaction V2'])]

  #Plot
  high_int_pdp_chart = alt.Chart(high_int_df_long, title = 'Plot 6: Three-Way PDPs').mark_line().encode(
  x = alt.X('X'),
  y = alt.Y('Y', title = 'Y (Centered)', scale = alt.Scale(domain = (ymin, ymax))),
  #color = alt.Color('Feature:N', scale = alt.Scale(domain = color_dom, range = color_ran), legend = None),
  detail = 'Feature:N',
  strokeDash = alt.StrokeDash('V2 Level',
                              legend = alt.Legend(orient='none', legendX = w + 10, legendY = 0.05*h, title='V2 Level',
                              symbolStrokeColor = 'black', symbolSize = 400)),
  color = alt.Color('V3 Level',
                    legend = alt.Legend(orient='none', legendX = w + 10, legendY = 0.35*h, title = 'V3 Level',
                              symbolSize = 400))
  ).properties(width = w, height = h)

  high_int_pdp_chart = high_int_pdp_chart.transform_filter(feature_selector1_click) #feature 1
  high_int_pdp_chart = high_int_pdp_chart.transform_filter(feature_selector3_click) #feature 2
  high_int_pdp_chart = high_int_pdp_chart.transform_filter(feature_selector4_click) #feature 3

  runningtime_PD3RIVER += timeit.default_timer() - timestamp_PD3RIVER
  timestamp_FEATURE_IMP = timeit.default_timer()

  final_plot =alt.vconcat(
    alt.hconcat(decomp_chart, pdp_chart).resolve_scale(color='independent'),
    alt.hconcat(int_chart, pdp_twoway_plot).resolve_scale(color='independent', strokeDash = 'independent'),
    #high_int_chart,
    alt.hconcat(high_int_chart, high_int_pdp_chart).resolve_scale(color='independent', strokeDash='independent')
  ).resolve_scale(color='independent', strokeDash='independent'
  ).configure_axis(labelFontSize=fontsize,titleFontSize=fontsize
                  ).configure_title(fontSize=fontsize).configure_view(stroke=None)


  runningtime_FEATURE_IMP += timeit.default_timer() - timestamp_FEATURE_IMP

  runtime_dict = {'Feature imp runtime': runningtime_FEATURE_IMP,
                  'PDP runtime': runningtime_PDP,
                  'PD2ICE runtime': runningtime_PD2ICE,
                  'PD2RIVER runtime': runningtime_PD2RIVER,
                  'PD3ICE runtime': runningtime_PD3ICE,
                  'PD3RIVER runtime': runningtime_PD3RIVER}

  #return {'plot': final_plot}
  #Uncomment below if you need to debug
  return {'plot': final_plot,
          'interactivity': _plot_feature_selectors,
          'pdp_df_oneway': pdp_df_combined,
          'pdp_df_twoway': pdp_df_twoway, 'int_df': int_df,
          'orig_int_df': orig_int_df,
          'feature_ranks': ranked,
          'graph_df': graph_df_full,'pdp_ice_list': pdp_ice_list,
           'runtime_dict': runtime_dict,
          'threeway_int_tuples': int_tuple_list,
          'pdp_df_threeway': high_int_df_long,
          'higher_order_df': high_int_df_long_3scores,
          'h2_filter2_list': h2_filter2_list,
          'h2_weight2_list': h2_weight2_list,
          }
