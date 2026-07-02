import numpy as np
import shap
from shap import KernelExplainer
from functools import partial


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

    return pipeline.predict_proba(X_reconstructed)[:,0]


def KernelShap(X_train,X_test,pipeline,baseline=None, multi_baseline = False, n_samples=2000):

    all_shap_values = []
    n_trials, n_channels, n_times = X_train.shape
    
    if baseline is None :
        baseline = np.mean(X_train,axis=0)
        
    predict_fn = stable_predict_multi is multi_baseline else stable_predict

    for run_index in range(X_test.shape[0]):
        
        stable_predict_frozen = partial(stable_predict, n_times=n_times, n_channels=n_channels, 
                                        current_run_signal=X_test[run_index] , 
                                        random_reference_signal=**({"background_signals": baseline} if multi_baseline else {"random_reference_signal": baseline}), pipeline=pipeline)
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



def KernelShapDeep(X_train,X_test,model,n_samples=2000):
    X_train = X_train.detach().numpy()
    X_test  = X_test.detach().numpy()
    n_trials, n_channels, n_times = X_train.shape
    random_reference_signal = np.mean(X_train, axis=0)

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
            else :
                pipeline.fit(X_train,y_train)
                shap_values = KernelShap(X_train,X_test, pipeline, n_samples=1000)
            all_shap_values.append(shap_values)
            all_scores.append(pipeline.score(X_test,y_test))

        return all_shap_values, all_scores
