####RGID FUNCTIONS

#These functions moved to Github on 12/4/2023
def neg_auc(y_true, y_pred):
  return -1 * roc_auc_score(y_true, y_pred)
def rmse(y_true, y_pred):
  return np.sqrt(np.mean((y_true - y_pred)**2))
def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)
def mae(y_true, y_pred):
  return np.mean(np.abs(y_true - y_pred))

def retrieve_quantiles(w):
  return np.cumsum(np.array([0]+list(w))/2 + np.array(list(w)+[0]) / 2)[0:-1]

def threeway_h2(f1_ice, f2_ice, f3_ice, f1_filters, f2_filters, f3_filters,
                f1_weights, f2_weights, f3_weights, grid_size,
               idf, f1, f2, f3, fnames, adjust_threeway = True, missing_threshold  = 0.25): #only considers three-way interactions where <25% of
               #value combinations are missing/not observed. This is because one-hot encoded variables from the same categorical variable
               #will be missing for 25% of their value combinations (won't have 1/1, but will have 0/0, 0/1, and 1/0), so this argument
               #prevents such one-hot encoded interactions from being surfaced

    max_missingness = 0

    grid_size1, grid_size2, grid_size3 = f1_ice.shape[0], f2_ice.shape[0], f3_ice.shape[0]
      #PDP version 1
    f1_zero_counts = np.any([(f2 & f3).sum() > 0 for f2 in f2_filters for f3 in f3_filters])
    if f1_zero_counts:
      pdp3_f1_f2f3 = np.concatenate(
      [np.array(
          [np.sum(f1_ice[:,f2 & f3] * ((w2*w3)[f2 & f3]).reshape(1,-1),axis=1) / np.max([np.sum((w2*w3)[f2 & f3]),1e-5]) for f2, w2 in zip(f2_filters, f2_weights)]).T.reshape(grid_size1,grid_size2,1) for f3,w3 in zip(f3_filters,f3_weights)], axis = 2)
      max_missingness = np.max([max_missingness, np.mean(pdp3_f1_f2f3==0)])
      pdp3_f1_f2f3[pdp3_f1_f2f3==0] = np.mean(pdp3_f1_f2f3[pdp3_f1_f2f3!=0])
    else:
      pdp3_f1_f2f3 = np.concatenate(
      [np.array(
        [np.average(f1_ice[:,f2 & f3], axis = 1, weights = (w2*w3)[f2 & f3]
                    ) for f2, w2 in zip(f2_filters, f2_weights)]).T.reshape(grid_size1,grid_size2,1) for f3,w3 in zip(f3_filters,f3_weights)], axis = 2)
      #PDP version 2
    f2_zero_counts = np.any([(f1 & f3).sum() > 0 for f1 in f1_filters for f3 in f3_filters])
    if f2_zero_counts:
      pdp3_f2_f1f3 = np.concatenate(
        [np.array(
            [np.sum(f2_ice[:,f1 & f3] * ((w1*w3)[f1&f3]).reshape(1,-1),axis = 1) / np.max([np.sum((w1*w3)[f1&f3]),1e-5]) for f1, w1 in zip(f1_filters,f1_weights)]).reshape(grid_size1,grid_size2,1) for f3, w3 in zip(f3_filters, f3_weights)],axis = 2)
      max_missingness = np.max([max_missingness, np.mean(pdp3_f2_f1f3==0)])
      pdp3_f2_f1f3[pdp3_f2_f1f3==0] = np.mean(pdp3_f2_f1f3[pdp3_f2_f1f3!=0])
    else:
      pdp3_f2_f1f3 = np.concatenate(
        [np.array(
            [np.average(f2_ice[:,f1 & f3], axis = 1, weights = (w1*w3)[f1 & f3]
                      ) for f1, w1 in zip(f1_filters,f1_weights)]).reshape(grid_size1,grid_size2,1) for f3, w3 in zip(f3_filters, f3_weights)],axis = 2)
      #PDP version 3
    f3_zero_counts = np.any([(f1 & f2).sum() > 0 for f1 in f1_filters for f2 in f2_filters])
    if f3_zero_counts:
      pdp3_f3_f1f2 = np.array(
        [np.array(
            [np.sum(f3_ice[:, f1 & f2] * ((w1*w2)[f1&f2]).reshape(1,-1), axis = 1)/np.max([np.sum((w1*w2)[f1&f2]),1e-5]) for f2, w2 in zip(f2_filters, f2_weights)]
            ) for f1, w1 in zip(f1_filters, f1_weights)]
      )
      max_missingness = np.max([max_missingness, np.mean(pdp3_f3_f1f2==0)])
      pdp3_f3_f1f2[pdp3_f3_f1f2==0] = np.mean(pdp3_f3_f1f2[pdp3_f3_f1f2!=0])
    else:
      pdp3_f3_f1f2 = np.array(
        [np.array(
            [np.average(f3_ice[:, f1 & f2], axis = 1, weights = (w1*w2)[f1 & f2]) for f2, w2 in zip(f2_filters, f2_weights)]
            ) for f1, w1 in zip(f1_filters, f1_weights)]
      )

    if pdp3_f1_f2f3.shape != (grid_size1,grid_size2,grid_size3):
      print('PDP3 v1 has shape mismatch')
    if pdp3_f2_f1f3.shape != (grid_size1,grid_size2,grid_size3):
      print('PDP3 v1 has shape mismatch')
    if pdp3_f3_f1f2.shape != (grid_size1,grid_size2,grid_size3):
      print('PDP3 v1 has shape mismatch')

            #Average the PDPs
    pdp3_final = np.average(
        np.concatenate([np.expand_dims(pdp3_f1_f2f3,0),np.expand_dims(pdp3_f2_f1f3,0),np.expand_dims(pdp3_f3_f1f2,0)],axis=0
        ), axis = 0)
        #weights = np.concatenate([np.expand_dims(pdp3_f1_f2f3_COUNTS,0),np.expand_dims(pdp3_f2_f1f3_COUNTS,0),np.expand_dims(pdp3_f3_f1f2_COUNTS,0)],axis=0)

    pdp3_final = pdp3_final - pdp3_final.mean()
    pdp3_f2f3, pdp3_f1f3, pdp3_f1f2 = [np.expand_dims(np.mean(pdp3_final, axis = a),a) for a in [0,1,2]]
    pdp3_f1, pdp3_f2, pdp3_f3 = [np.expand_dims(np.mean(pdp3_final, axis = tup), axis = tup) for tup in [(1,2),(0,2),(0,1)]]

    full_h2 =  np.sum(
     (pdp3_final - pdp3_f1f2 - pdp3_f2f3 - pdp3_f1f3 + pdp3_f1 + pdp3_f2 + pdp3_f3)**2
    ) / np.sum(pdp3_final**2)

    cor12 = np.corrcoef(pdp3_f1_f2f3.reshape(-1),pdp3_f2_f1f3.reshape(-1))[0,1]
    cor23 = np.corrcoef(pdp3_f2_f1f3.reshape(-1),pdp3_f3_f1f2.reshape(-1))[0,1]
    cor13 = np.corrcoef(pdp3_f1_f2f3.reshape(-1),pdp3_f3_f1f2.reshape(-1))[0,1]
    avg_cor = np.mean([cor12,cor23,cor13])

    if (max_missingness < missing_threshold):
        #resid_h2 =  np.float(full_h2)
        pfull_h2_12= np.sum((pdp3_final.mean(axis = 2) - pdp3_final.mean(axis = 2).mean(axis=0).reshape(1,-1) - pdp3_final.mean(axis = 2).mean(axis=1).reshape(-1,1) )**2) / np.sum(pdp3_final.mean(axis = 2)**2)
        pfull_h2_23 = np.sum((pdp3_final.mean(axis = 0) - pdp3_final.mean(axis = 0).mean(axis=0).reshape(1,-1) - pdp3_final.mean(axis = 0).mean(axis=1).reshape(-1,1) )**2) / np.sum(pdp3_final.mean(axis = 0)**2)
        pfull_h2_13 = np.sum((pdp3_final.mean(axis = 1) - pdp3_final.mean(axis = 1).mean(axis=0).reshape(1,-1) - pdp3_final.mean(axis = 1).mean(axis=1).reshape(-1,1) )**2) / np.sum(pdp3_final.mean(axis = 1)**2)

        if fnames is None:
          real_h2_12 = idf[(idf['Feature'] == 'X'+str(f1)) & (idf['Feature 2'] == 'X'+str(f2))]['Diff'].squeeze()
          real_h2_23 = idf[(idf['Feature'] == 'X'+str(f2)) & (idf['Feature 2'] == 'X'+str(f3))]['Diff'].squeeze()
          real_h2_13 = idf[(idf['Feature'] == 'X'+str(f1)) & (idf['Feature 2'] == 'X'+str(f3))]['Diff'].squeeze()
        else:
          real_h2_12 = idf[(idf['Feature'] == fnames[f1]) & (idf['Feature 2'] == fnames[f2])]['Diff'].squeeze()
          real_h2_23 = idf[(idf['Feature'] == fnames[f2]) & (idf['Feature 2'] == fnames[f3])]['Diff'].squeeze()
          real_h2_13 = idf[(idf['Feature'] == fnames[f1]) & (idf['Feature 2'] == fnames[f3])]['Diff'].squeeze()

        pfull_vec = np.clip([pfull_h2_12, pfull_h2_23, pfull_h2_13], a_min = 0.00000001, a_max = None)
        avg_pdp_h2 = np.exp(np.mean(np.log(pfull_vec)))
        h2_vec = np.clip([real_h2_12, real_h2_23, real_h2_13], a_min = 0.00000001, a_max = None)
        avg_real_h2 = np.exp(np.mean(np.log(h2_vec)))

        if adjust_threeway == True:
          resid_h2 = (full_h2 / avg_pdp_h2) * avg_real_h2
        else:
          resid_h2 = full_h2

    else:
      resid_h2 = 0
    return resid_h2, avg_cor, avg_real_h2 / avg_pdp_h2

def get_neighborhood_points(x, q_central, neighborhood_size):
  qmin = np.max([q_central - neighborhood_size/2, 0])
  qmax = np.min([q_central + neighborhood_size/2, 1])
  if np.isclose(np.quantile(x, qmin), np.quantile(x, qmax)):
    filt = (x == np.quantile(x, qmin))
  else:
    filt = (x > np.quantile(x, qmin)) & (x <= np.quantile(x, qmax))
  if np.sum(filt) == 0: #try including lower bound
    filt = (x >= np.quantile(x, qmin)) & (x <= np.quantile(x, qmax))

  if np.all(filt==True):
    idx = np.random.choice(np.arange(len(x)), size = int(np.floor(neighborhood_size * len(x))), replace = False)
    filt = np.array([False] * len(x))
    filt[idx] = True
  return filt

def logit_catch_singular(y, X_train, X):
  from statsmodels.tools.sm_exceptions import PerfectSeparationError

  try:
      # Try fitting with statsmodels
      sm_model = sm.Logit(y, X_train).fit(disp=0)
      return sm_model.predict(X)

  except (np.linalg.LinAlgError, PerfectSeparationError) as e:
      print(f"Caught propensity regression with error: {str(e)}")
      print("Switching to ridge logistic regression...")
      sm_model = LogisticRegression(penalty='l2',fit_intercept=False).fit(X_train, y)
      return sm_model.predict_proba(X)[:,1]


def propensity_filters_and_weights(f, pdp_w_list, pdp2_band_width,
   X,pdp_ips_trim_q, pdp3_band_width = None, propensity_samples = 1000,
  run_threeway = False):

  X = np.array(X)

  X_lr = (np.array(X) - np.array(X).mean(axis = 0).reshape(1,-1)) / np.array(X).std(axis=0).reshape(1,-1) #standardize so that LR expected value is always 50%. This reduces variance in weights
  if propensity_samples >= X_lr.shape[0]:
    X_lr_train = np.array(X_lr)
    sample_index = np.arange(X_lr.shape[0])
  else:
    np.random.seed(propensity_samples)
    sample_index = np.random.choice(range(X_lr.shape[0]), size = propensity_samples)
    X_lr_train = np.array(X_lr)[sample_index,:] #note that X_lr_train is standardized based on X_lr (which may be larger), but that's okay

  regularize = False

  if np.linalg.matrix_rank(np.delete(X_lr_train, f, axis = 1)) < np.delete(X_lr_train, f, axis = 1).shape[1]:
    regularize == True

  f_quantiles = np.array(retrieve_quantiles(pdp_w_list[f]))

  f_quantile_bands2 = [(np.max([q - pdp2_band_width/2,0]), np.min([q + pdp2_band_width/2,1])) for q in f_quantiles]

  if run_threeway == True:
    f_quantile_bands3 = [(np.max([q - pdp3_band_width/2,0]), np.min([q + pdp3_band_width/2,1])) for q in f_quantiles]

  #If variable has more values than weights (meaning more values than grid_size), use the standard formula
  unique = np.unique(X[:,f])
  if len(unique) > len(pdp_w_list[f]): #i.e., if there are enough values that cutpoints() shortened to grid_size
    f_filters2 = [ (X[:,f] > np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
        for (q1, q2) in f_quantile_bands2 ]
    if np.any([f.sum()==0 for f in f_filters2]): #include lower bound if too few values
        f_filters2 = [ (X[:,f] >= np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
        for (q1, q2) in f_quantile_bands2 ]

    if run_threeway == True:
      f_filters3 = [(X[:,f] > np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
          for (q1, q2) in f_quantile_bands3]
      if np.any([f.sum()==0 for f in f_filters3]): #include lower bound if too few values
          f_filters3 = [ (X[:,f] >= np.quantile(X[:,f], q1)) & (X[:,f] <= np.quantile(X[:,f], q2)) \
          for (q1, q2) in f_quantile_bands3 ]

  else: #use X's unique values if there are fewer unique values than weights
    f_filters2 = [ X[:,f]==u for u in unique]
    if run_threeway == True:
      f_filters3 = [ X[:,f]==u for u in unique]

  if run_threeway == True:
    if (np.any([np.std(f)==0 for f in f_filters2])) | (np.any([np.std(f)==0 for f in f_filters3])):
      print(f)
      print(pdp_w_list[f])
      print(f_quantile_bands2)
      print(f_quantile_bands3)
      print('Replacing quantile-based propensity scores with exception-handlind formula')
      f_filters2 = [get_neighborhood_points(X[:,f], q, pdp2_band_width) for q in f_quantiles]
      f_filters3 = [get_neighborhood_points(X[:,f], q, pdp3_band_width) for q in f_quantiles]
  if run_threeway == False:
    if (np.any([np.std(f)==0 for f in f_filters2])):
      print(f)
      print(pdp_w_list[f])
      print(f_quantile_bands2)
      print('Replacing quantile-based propensity scores with exception-handlind formula')
      f_filters2 = [get_neighborhood_points(X[:,f], q, pdp2_band_width) for q in f_quantiles]

  #Now compute propensities
  if regularize == True:
    f_propensities2 = [LogisticRegression(penalty='l2',fit_intercept=False).fit(
          np.delete(X_lr_train, f, axis = 1), filt[sample_index]).predict_proba(np.delete(X_lr, f, axis = 1))[:,1] for filt in f_filters2]
    if run_threeay == True:
      f_propensities3 = [LogisticRegression(penalty='l2',fit_intercept=False).fit(
            np.delete(X_lr_train, f, axis = 1), filt[sample_index]).predict_proba(np.delete(X_lr, f, axis = 1))[:,1] for filt in f_filters3]

  else:
    if X_lr_train.shape[0] < X_lr.shape[0]: #if using different train and test matrices
      f_propensities2 = [logit_catch_singular(filt[sample_index], X_train = np.delete(X_lr_train, f, axis = 1), X = np.delete(X_lr, f, axis = 1)) for filt in f_filters2]
      if run_threeway == True:
        f_propensities3 = [logit_catch_singular(filt[sample_index], X_train = np.delete(X_lr_train, f, axis = 1), X = np.delete(X_lr, f, axis = 1)) for filt in f_filters3]
    else:
      f_propensities2 = [logit_catch_singular(filt, X_train = np.delete(X_lr, f, axis = 1), X = np.delete(X_lr, f, axis = 1)) for filt in f_filters2]
      if run_threeway == True:
        f_propensities3 = [logit_catch_singular(filt, X_train = np.delete(X_lr, f, axis = 1), X = np.delete(X_lr, f, axis = 1)) for filt in f_filters3]

  f_weights2 = [np.clip(1/np.clip(fp,a_max=None,a_min=1e-4), a_min = None, a_max = np.quantile(1/np.clip(fp,a_max=None,a_min=1e-4), pdp_ips_trim_q)) for fp in f_propensities2]
  if run_threeway == True:
    f_weights3 = [np.clip(1/np.clip(fp,a_max=None,a_min=1e-4), a_min = None, a_max = np.quantile(1/np.clip(fp,a_max=None,a_min=1e-4), pdp_ips_trim_q)) for fp in f_propensities3]
    return f_filters2, f_weights2, f_filters3, f_weights3
  if run_threeway == False:
    return f_filters2, f_weights2

def pdp_propensity_filters_and_weights(f, pdp_high_filter, pdp_low_filter, X, pdp_ips_trim_q, propensity_samples = 1000):
  #Computing 2PDQUIVER propensities outside their own function due to simplicity
  X_lr = (np.array(X) - np.array(X).mean(axis = 0).reshape(1,-1)) / np.array(X).std(axis=0).reshape(1,-1) #standardize so that LR expected value is always 50%. This reduces variance in weights

  if propensity_samples >= X_lr.shape[0]:
    X_lr_train = np.array(X_lr)
    sample_index = np.arange(X_lr.shape[0])
  else:
    np.random.seed(propensity_samples)
    sample_index = np.random.choice(range(X_lr.shape[0]), size = propensity_samples)
    X_lr_train = np.array(X_lr)[sample_index,:] #note that X_lr_train is standardized based on X_lr (which may be larger), but that's okay

  regularize = False
  if np.linalg.matrix_rank(np.delete(X_lr_train, f, axis = 1)) < np.delete(X_lr_train, f, axis = 1).shape[1]:
    regularize = True

  #Now compute propensities
  if regularize == True: #fit from X_lr_train, predict based on X_lr (which may be same matrix if X.shape[0] < propensity_samples)
    high_propensities = LogisticRegression(penalty='l2',fit_intercept=False).fit(
        np.delete(X_lr_train, f, axis = 1),
        pdp_high_filter[sample_index,f] #use sample_index to subset y (which may just be all rows)
        ).predict_proba(np.delete(X_lr, f, axis = 1))[:,1]
    low_propensities = LogisticRegression(penalty='l2',fit_intercept=False).fit(
        np.delete(X_lr_train, f, axis = 1),
        pdp_low_filter[sample_index,f] #use sample_index to subset y (which may just be all rows)
        ).predict_proba(np.delete(X_lr, f, axis = 1))[:,1]

  else:
    if X_lr_train.shape[0] < X_lr.shape[0]: #if using different train matrix from X, fit from X_lr_train and predict from X_lr
      #high_propensities = sm.Logit(pdp_high_filter[sample_index,f], np.delete(X_lr_train, f, axis = 1)).fit(disp=0).predict(np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being above its qth quantile
      #low_propensities = sm.Logit(pdp_low_filter[sample_index,f], np.delete(X_lr_train, f, axis = 1)).fit(disp=0).predict(np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being below its qth quantile

      high_propensities = logit_catch_singular(pdp_high_filter[sample_index,f], X_train = np.delete(X_lr_train, f, axis = 1), X = np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being above its qth quantile
      low_propensities = logit_catch_singular(pdp_low_filter[sample_index,f], X_train = np.delete(X_lr_train, f, axis = 1), X = np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being below its qth quantile

    else: #if X_lr and X_lr_train are identical, just fit/predict from X_lr
      #high_propensities = sm.Logit(pdp_high_filter[:,f], np.delete(X_lr, f, axis = 1)).fit(disp=0).predict() #get propensities for that entire column being above its qth quantile
      #low_propensities = sm.Logit(pdp_low_filter[:,f], np.delete(X_lr, f, axis = 1)).fit(disp=0).predict() #get propensities for that entire column being below its qth quantile

      high_propensities = logit_catch_singular(pdp_high_filter[:,f], X_train=np.delete(X_lr, f, axis = 1),X=np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being above its qth quantile
      low_propensities = logit_catch_singular(pdp_low_filter[:,f], X_train=np.delete(X_lr, f, axis = 1),X=np.delete(X_lr, f, axis = 1)) #get propensities for that entire column being below its qth quantile

  high_weights = np.clip(1/np.clip(high_propensities,a_max=None,a_min=1e-4), a_min=None, a_max = np.quantile(1/np.clip(high_propensities,a_max=None,a_min=1e-4), pdp_ips_trim_q)) #convert to weights
  low_weights = np.clip(1/np.clip(low_propensities,a_max=None,a_min=1e-4), a_min=None, a_max = np.quantile(1/np.clip(low_propensities,a_max=None,a_min=1e-4), pdp_ips_trim_q)) #convert to weights

  return high_weights, low_weights


 #Write and run function getting cutpoints for feature f
def cutpoints(var, grid_size): #cutpoints are MIDPOINTS of evenly divided segments
  if len(np.unique(var)) <= grid_size:
    values = np.sort(np.unique(var))
    weights = np.array([np.mean(var == v) for v in values])
    df_final = pd.DataFrame({'values': values, 'weights': weights})
  else:
    width = 1 / grid_size
    quantiles = np.arange(0, 1, width) + width/2
    weights = np.ones(grid_size) / grid_size
    values = np.array([np.quantile(var, q, method = 'inverted_cdf') for q in quantiles])
    values = np.round(values, 3) #rounding allows us to lump similar quantiles together
    df = pd.DataFrame({'values': values, 'weights': weights})
    df_final = df.groupby('values', as_index = False)['weights'].sum()
  return df_final


#Create feature-specific PDP function
def feature_importance_scores(X, y, model, f, metric, grid_size = 20,
                              int_only = False,
                              predict_proba = False, X_residuals = None, interaction_quantiles = (0.25, 0.75)):

  #Choose a continuous feature, then make predictions for all imputed values of that feature
  if isinstance(X, pd.core.frame.DataFrame):
    is_df = True
    fnames = X.columns
  else:
    is_df = False
  #Identify grid points for variable f
  c = cutpoints(X.iloc[:,f], grid_size = grid_size) if is_df == True else cutpoints(X[:,f], grid_size = grid_size)
  weights = c['weights'].values
  values = c['values'].values

  #Print number of unique values
  n_unique = len(np.unique(X.iloc[:,f])) if is_df == True else len(np.unique(X[:,f]))
  if n_unique < len(values):
    print('Number of quantile points greater than number of unique values')

  #Create two-way PDP matrix
  pdp = np.zeros((len(values), X.shape[0]))

  #Loop through all cutpoints values
  X_impute = X.copy(deep = True) if is_df == True else np.array(X)

  for v, i in zip(values, range(len(values))):
    if is_df == True:
     X_impute.iloc[:,f] = v
    else:
     X_impute[:,f] = v
    _yhat = model.predict(X = X_impute) if predict_proba == False else model.predict_proba(X= X_impute)[:,1] #get '1' class probabilities

    pdp[i, :] = _yhat.reshape(-1)
    #pdp[i, :] = _yhat_results[i].reshape(-1)

    #Create one-way PDP's by averaging across both axes
  #if conditional==False:
  f_not_pdp = np.average(pdp, axis = 0, weights = weights)
  f_pdp = pdp.mean(axis=1).reshape(-1)
  f_pdp_interp = np.interp(X.iloc[:,f], values, f_pdp) if is_df == True else np.interp(X[:,f], values, f_pdp)

    #Center both pdps
  f_not_pdp = f_not_pdp - f_not_pdp.mean()
  f_pdp_interp = f_pdp_interp - f_pdp_interp.mean()

  #Add linear trendline to PDP
  if is_df==True:
    lm_model = sm.OLS(f_pdp_interp.reshape(-1), sm.add_constant(X.iloc[:,f]))
    results = lm_model.fit()
    linear_pdp_component = results.params[0] + results.params[1] * X.iloc[:,f] #has n entries

  else:
    lm_model = sm.OLS(f_pdp_interp.reshape(-1), sm.add_constant(X[:,f]))
    results = lm_model.fit()
    linear_pdp_component = results.params[0] + results.params[1] * X[:,f] #has n entries

  #Calculate feature contributions to overall predictions
  yhat_uncentered = model.predict(X) if predict_proba == False else model.predict_proba(X)[:,1]
  yhat_mean = yhat_uncentered.mean()
  yhat = yhat_uncentered - yhat_mean

  f_int = yhat - f_pdp_interp - f_not_pdp #n entries
  f_pdp_linear = linear_pdp_component #n entries
  f_pdp_nonlinear = f_pdp_interp - f_pdp_linear #n entries

  #Compute prediction components
  whole_pred = f_not_pdp + f_pdp_linear + f_pdp_nonlinear + f_int
  none_pred = f_not_pdp
  h2 = np.sum((yhat - f_pdp_interp - f_not_pdp)**2) / np.sum(yhat**2)

  #Check if feature f can be perfectly predicted from other features
  feature_pred_ols = sm.OLS(X.iloc[:,f], sm.add_constant(X.drop(X.columns[f], axis = 1))).fit() if is_df == True else sm.OLS(X[:,f],
                                                                                    sm.add_constant(np.delete(X, f, axis = 1))).fit()
  if feature_pred_ols.rsquared >= 0.99:
    nolinear_pred = f_not_pdp + f_pdp_nonlinear + f_int
    no_nonlinear_pred = f_not_pdp + f_pdp_linear + f_int
    no_int_pred = f_not_pdp + f_pdp_linear + f_pdp_nonlinear

  #NOTE: This section helps recover some of the covariance lost when the interaction term is omitted.
  #This attenuates the intreaction estimate when predictors are correlated (but no interaction is actually present)
  #whole_pred = sm.OLS(y, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_pdp_nonlinear,f_int],axis=1))).fit().predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_pdp_nonlinear,f_int],axis=1)))
  else:
    nolinear_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_nonlinear,f_int],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_nonlinear,f_int],axis=1)))

    no_nonlinear_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_int],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_int],axis=1)))

    nomarginal_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_interp],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_interp],axis=1)))

    no_int_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear + f_pdp_nonlinear],axis=1))).fit(
    ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear + f_pdp_nonlinear],axis=1)))

  # #Compute MSE's
  # whole_pred = f_not_pdp + f_pdp_linear + f_pdp_nonlinear + f_int
  # none_pred = f_not_pdp

  # h2 = np.sum((yhat - f_pdp_interp - f_not_pdp)**2) / np.sum(yhat**2)

  # #NOTE: This section helps recover some of the covariance lost when the interaction term is omitted.
  # #This attenuates the intreaction estimate when predictors are correlated (but no interaction is actually present)
  # nolinear_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_nonlinear,f_int],axis=1))).fit(
  # ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_nonlinear,f_int],axis=1)))

  # no_nonlinear_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_int],axis=1))).fit(
  # ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear,f_int],axis=1)))

  # no_int_pred = sm.OLS(whole_pred, sm.add_constant(np.stack([f_not_pdp,f_pdp_linear + f_pdp_nonlinear],axis=1))).fit(
  # ).predict(sm.add_constant(np.stack([f_not_pdp,f_pdp_linear + f_pdp_nonlinear],axis=1)))

  #Calculate score
  if metric=='rmse':
    metric = rmse
  if metric=='mse':
    metric = mse

  #Add yhat_mean back to 'un mean center' the predictions
  score = metric(y, yhat_uncentered)
  score_no_linear = metric(y, nolinear_pred + yhat_mean)
  score_no_nonlinear = metric(y, no_nonlinear_pred + yhat_mean)
  score_no_marginal = metric(y, nomarginal_pred + yhat_mean)

  score_no_int = metric(y, no_int_pred + yhat_mean)
  score_none = metric(y, none_pred + yhat_mean)

  linear_score = score_no_linear - score
  nonlinear_score = score_no_nonlinear - score
  marginal_score = score_no_marginal - score
  none_score = score_none - score
  #int_score = score_no_int - score - OLD METHOD
  if int_only == False:
    int_score = none_score - linear_score - nonlinear_score
  else:
    int_score = none_score - marginal_score

  if int_only == False:
    return linear_score, nonlinear_score, int_score, none_score, values, \
      f_pdp, f_not_pdp, f_pdp_interp, pdp, h2, weights
      #f_pdp is not mean centered
  else:
    return marginal_score, int_score, none_score, values, \
      f_pdp, f_not_pdp, f_pdp_interp, pdp, h2, weights



def egive_center_pdp(pdp_df):
  full_df = pdp_df.copy(deep = True)
  feature_vars = ['Feature','Interaction V1','Interaction V2']

  #Initial mean centering
  full_df['Y'] = full_df['Y'] - full_df.groupby(
      feature_vars + ['V2 Level','V3 Level']
      )['Y'].transform('mean')
  return full_df

def egive_append_threeway_scores(pdp_df):
  full_df = pdp_df.copy(deep = True)
  feature_vars = ['Feature','Interaction V1','Interaction V2']

  full_df = egive_center_pdp(full_df)

  full_df['pdp_avg'] = full_df.groupby(feature_vars + ['X'])['Y'].transform('mean')
  full_df['pdp_c'] = full_df['Y'] - full_df['pdp_avg']
  full_df['pdp_x2'] = full_df.groupby(feature_vars + ['X','V2 Level'])['Y'].transform('mean')
  full_df['pdp_x3'] = full_df.groupby(feature_vars + ['X','V3 Level'])['Y'].transform('mean')
  full_df['pdp_x2_c'] = full_df['pdp_x2'] - full_df['pdp_avg']
  full_df['pdp_x3_c'] = full_df['pdp_x3'] - full_df['pdp_avg']

  full_df['numer'] = (full_df['pdp_c'] - full_df['pdp_x2_c'] - full_df['pdp_x3_c'])**2
  #full_df['denom'] = (full_df['pdp_c'])**2

  summ_df = full_df.groupby(feature_vars + ['X'], as_index = False)['numer'].sum()
  #summ_df['numer_denom'] = summ_df['numer'] / np.clip(summ_df['denom'], a_min = 0.0001, a_max = None)
  #summ_df = summ_df.groupby(feature_vars, as_index = False)['numer_denom'].min()
  summ_df = summ_df.groupby(feature_vars, as_index = False)['numer'].std()

  summ_df.columns = feature_vars + ['Threeway Score']
  pdp_df = pdp_df.merge(summ_df[feature_vars + ['Threeway Score']], how = 'left', on = feature_vars)
  #pdp_ranges = pdp_df.groupby(feature_vars)['Y'].transform('std') #- pdp_df.groupby(feature_vars)['Y'].transform('min')
  #pdp_df['Threeway Score'] = pdp_df['Threeway Score'] / pdp_ranges
  return pdp_df

