import numpy as np
import shap
import torch
from shap import KernelExplainer
from functools import partial
from sklearn.model_selection import train_test_split
from pyriemann.estimation import Covariances
from src.SPDNet.SPDNet import SPDNetBatchNorm, train_model
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig


def stable_predict_multi(mask_2d, n_channels, n_times, current_run_signal, background_signals, pipeline):
    n_simulations = mask_2d.shape[0]
    n_background = background_signals.shape[0]
    
    # On va stocker les prédictions moyennes pour chaque simulation
    predictions_accumulated = np.zeros(n_simulations)
    
    # Pour chaque signal de référence dans notre background dataset
    for bg_idx in range(n_background):
        X_reconstructed = np.zeros((n_simulations, n_channels, n_times))
        ref_signal = background_signals[bg_idx]
        
        for i in range(n_simulations):
            for ch in range(n_channels):
                if mask_2d[i, ch] > 0.5:
                    X_reconstructed[i, ch, :] = current_run_signal[ch, :]
                else:
                    X_reconstructed[i, ch, :] = ref_signal[ch, :]
                    
        # On accumule les probabilités de la classe 0
        predictions_accumulated += pipeline.predict_proba(X_reconstructed)[:, 0]
        
    # On renvoie la moyenne des prédictions à travers toutes les références
    return predictions_accumulated / n_background


def stable_predict(mask_2d,n_channels,n_times,current_run_signal,random_reference_signal,pipeline):
    # mask_2d a une forme (N_simulations, n_channels)
    n_simulations = mask_2d.shape[0]
    X_reconstructed = np.zeros((n_simulations, n_channels, n_times))
    
    for i in range(n_simulations):
        for ch in range(n_channels):
            # Si le masque SHAP dit 1, on prend le signal du run actuel
            if mask_2d[i, ch] > 0.5:
                X_reconstructed[i, ch, :] = current_run_signal[ch, :]
            else:
                X_reconstructed[i, ch, :] = random_reference_signal[ch, :]

    return pipeline.predict_proba(X_reconstructed)



def KernelShap(X_train,X_test,pipeline,baseline=None, multi_baseline = False, n_samples=2000):

    all_shap_values = []
    n_trials, n_channels, n_times = X_train.shape
    
    if baseline is None :
        baseline = np.mean(X_train,axis=0)
    
    #print(baseline)
        
    predict_fn = stable_predict_multi if multi_baseline else stable_predict

    for run_index in range(X_test.shape[0]):
        
        stable_predict_frozen = partial(stable_predict, n_times=n_times, n_channels=n_channels, 
                                        current_run_signal=X_test[run_index] , 
                                        **({"background_signals": baseline} if multi_baseline else {"random_reference_signal": baseline}), pipeline=pipeline)
        explainer = KernelExplainer(stable_predict_frozen, np.zeros((1, n_channels)),seed=run_index)
        
        shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)

        all_shap_values.append(shap_values)

    return all_shap_values


def stable_predict_deep(mask_2d,n_channels,n_times,current_run_signal,random_reference_signal,model):
    # mask_2d a une forme (N_simulations, n_channels)
    n_simulations = mask_2d.shape[0]
    X_reconstructed = np.zeros((n_simulations, n_channels, n_times))
    
    for i in range(n_simulations):
        for ch in range(n_channels):
            # Si le masque SHAP dit 1, on prend le signal du run actuel
            if mask_2d[i, ch] > 0.5:
                X_reconstructed[i, ch, :] = current_run_signal[ch, :]
            else:
                X_reconstructed[i, ch, :] = random_reference_signal[ch, :]
                
    X_reconstructed = torch.tensor(
    X_reconstructed,
    dtype=torch.float32

)
    with torch.no_grad():
        out = torch.softmax(model(X_reconstructed), dim=1)
    
    return out.detach().numpy()



def KernelShapDeep(X_train,X_test,model,baseline, n_samples=2000):
    cov_layer = CovLayer()
    X_train = X_train.detach().numpy()
    
    X_test  = X_test.detach().numpy()
    n_trials, n_channels, n_times = X_train.shape
    random_reference_signal = baseline
    all_shap_values = []
    for run_index in range(X_test.shape[0]):
        current_run_signal = X_test[run_index] 
        
        stable_predict_frozen = partial(stable_predict_deep, n_times=n_times, n_channels=n_channels, 
                                        current_run_signal=current_run_signal, 
                                        random_reference_signal=random_reference_signal, model=model)
        explainer = KernelExplainer(stable_predict_frozen, np.zeros((1, n_channels)),seed=run_index)
        
        shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)

        all_shap_values.append(shap_values)

    return all_shap_values

    
def compute_shapley(X,y, n_splits, model_config, deep= False):
    if deep:

        all_shap_values = []
        all_scores = []

        for i in range(n_splits):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=0.8,
                stratify=y,
                random_state=i
            )

            covariance = Covariances()
            covs = covariance.fit_transform(X_train)

            variances = np.array([
                [covs[j,k,k] for k in range(covs.shape[1])]
                for j in range(covs.shape[0])
            ])
            print(variances.shape)

            var_left = np.mean(variances[y_train==0], axis=0)
            var_right = np.mean(variances[y_train==1], axis=0)
            print(var_left.shape)
            var_baseline = (var_left + var_right) / 2
            print(var_baseline.shape)

            mean_signal = np.mean(X_train, axis=0)

            noise = np.random.normal(
                0,
                scale=np.sqrt(var_baseline[:, np.newaxis]),
                size=mean_signal.shape
            )

            baseline = mean_signal + noise

            X_train_t = torch.tensor(X_train).float()
            X_test_t = torch.tensor(X_test).float()

            y_train_t = torch.tensor(y_train).long()
            y_test_t = torch.tensor(y_test).long()

            model = SPDNetBatchNorm(
                **model_config,
                input_type="raw"
            ).float()


            train_model(
                model,
                X_train_t,
                y_train_t,
                X_test_t,
                y_test_t,
                epochs=200,
                lr=1e-3
            )

            shap_values = KernelShapDeep(
                X_train_t,
                X_test_t,
                model,
                baseline,
                n_samples=1000
            )
            all_shap_values.append(shap_values)
            all_scores.append((model(X_test_t).argmax(1) == y_test).float().mean().item())

    else : 
        pipeline = model_config
        all_shap_values = []
        all_scores = []
        for i in range (n_splits):
            print("Split",i)
            X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.8, stratify=y, random_state=i)
            covariance = Covariances()
            covs = covariance.transform(X_train)
            vars = np.array([[covs[j,i,i] for i in range (covs.shape[1])] for j in range(covs.shape[0])])
            var_left = np.mean(vars[y_train=='left_hand'], axis=0)
            var_right = np.mean(vars[y_train=='right_hand'], axis=0)
            print(var_left)
            print(var_right)
            var_baseline = (var_left + var_right) / 2
            mean_signal = np.mean(X_train, axis=0)
            noise = np.random.normal(0, scale=np.sqrt(var_baseline[:, np.newaxis]), size=mean_signal.shape)
            print("noise",noise)
            baseline = mean_signal + noise  
            #print(covariance.fit_transform(baseline))
            pipeline.fit(X_train,y_train)
            print(pipeline.predict_proba(baseline[np.newaxis, :, :]))
            shap_values = KernelShap(X_train,X_test, pipeline, baseline=baseline, n_samples=1000)
            all_shap_values.append(shap_values)
            all_scores.append(pipeline.score(X_test,y_test))

    return all_shap_values, all_scores

def compute_shapley_corr(X,y, n_splits, model_config, deep= False):
    if deep :
        all_shap_values = []
        all_scores = []

        for i in range(n_splits):
            print("Split", i)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, stratify=y, random_state=i
            )

            model = SPDNetBatchNorm(**model_config).float()
            train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=1e-3)
            model.eval()

            shap_values = KernelShapDeep(X_train, X_test, model, n_samples=2000)
            all_shap_values.append(shap_values)
            all_scores.append((model(X_test).argmax(1) == y_test).float().mean().item())

    else : 
        pipeline = model_config
        all_shap_values = []
        all_scores = []
        for i in range (n_splits):
            print("Split",i)
            X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.8, stratify=y, random_state=i)
            covariance = Covariances()
            covs = covariance.transform(X_train)
            vars = np.array([[covs[j,i,i] for i in range (covs.shape[1])] for j in range(covs.shape[0])])
            var_left = np.mean(vars[y_train=='left_hand'], axis=0)
            var_right = np.mean(vars[y_train=='right_hand'], axis=0)
            var_baseline = (var_left + var_right) / 2
            mean_signal = np.mean(X_train, axis=0)
            noise = np.random.normal(0, scale=np.sqrt(var_baseline[:, np.newaxis]), size=mean_signal.shape)
            baseline = mean_signal + noise  
            #print(covariance.fit_transform(baseline))
            pipeline.fit(X_train,y_train)
            print(pipeline.predict_proba(baseline[np.newaxis, :, :]))
            shap_values = KernelShap(X_train,X_test, pipeline, baseline=baseline, n_samples=1000)
            all_shap_values.append(shap_values)
            all_scores.append(pipeline.score(X_test,y_test))

        return all_shap_values, all_scores