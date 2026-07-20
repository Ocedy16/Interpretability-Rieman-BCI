import argparse
from src.data.dataset_config import DATASET_CONFIG
from moabb.paradigms import FilterBankMotorImagery
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from src.SPDNet.SPDNet import fit
from src.Visualization.topomap import plot_topomap, plot_pannel
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        required=True,
        help="Dataset à utiliser : BNCI2014_001, Dreyer2023C, Beetl2021_A"
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Nombre de splits (défaut: 10)"
    )

    parser.add_argument(
        "--n_perms",
        type=int,
        default=10,
        help="Nombre de permutations (défaut: 10)"
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    cfg = DATASET_CONFIG[args.dataset]
    dataset = cfg["dataset"]
    SENSORS = cfg["sensors"]
    visu_only = False
    SCORE_THRESHOLD = 0.75

    OUT_DIR = f'../../Results/Permutation/SPDNet/{args.dataset}/SansNormalisation/Modif_eeg'

    paradigm = FilterBankMotorImagery(filters=[[7, 35]],events={"left_hand": 1, "right_hand": 2})
    X, y, meta = paradigm.get_data(dataset)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model_config = {
        "n_chans": X.shape[1],
        "n_outputs": 2,
        "subspacedim": 16, 
        "bn" : None
    }

    if visu_only == False : 
        

        results_dic = fit(X, y_encoded, meta, model_config, n_iter=args.n_splits, n_perm=args.n_perms)

        filename = os.path.join(OUT_DIR, 'results_dic_norm.pkl')
        with open(filename,'wb') as f:
            pickle.dump(results_dic,f)

        
        all_results = []
        all_scores = []
        good_subjects_shap = []
        good_subjects_scores = []
        bad_subjects_shap = []
        bad_subjects_scores = []

        for subject in dataset.subject_list:    
            results = results_dic['across_times'][subject]['importance']
            scores = results_dic['across_times'][subject]['accuracy']
            all_results.append(np.mean(np.array(results),axis=0))
            all_scores.append(np.mean(np.array(scores),axis=0))

            if np.mean(np.array(scores),axis=0) >= SCORE_THRESHOLD : 
                good_subjects_shap.append(np.mean(np.array(results),axis=0))
                good_subjects_scores.append(np.mean(np.array(scores),axis=0))

            else: 
                bad_subjects_shap.append(np.mean(np.array(results),axis=0))
                bad_subjects_scores.append(np.mean(np.array(scores),axis=0))
            
        np.save(f'{OUT_DIR}/all_scores.npy', np.array(all_scores))  
        np.save(f'{OUT_DIR}/good_subjects.npy', np.array(good_subjects_shap))
        np.save(f'{OUT_DIR}/bad_subjects.npy', np.array(bad_subjects_shap))
        np.save(f'{OUT_DIR}/good_subjects_scores.npy', np.array(good_subjects_scores))
        np.save(f'{OUT_DIR}/bad_subjects_scores.npy', np.array(bad_subjects_scores))

        plot_pannel(all_results, dataset, SENSORS, all_scores, OUT_DIR,cbar_type='Permutation', suptitle = f"Feature Permutation Importance on {args.dataset}")

        

    good_subjects_shap = np.load(f'{OUT_DIR}/good_subjects.npy')
    bad_subjects_shap = np.load(f'{OUT_DIR}/bad_subjects.npy')
    good_subjects_scores = np.load(f'{OUT_DIR}/good_subjects_scores.npy')
    bad_subjects_scores = np.load(f'{OUT_DIR}/bad_subjects_scores.npy')



    v_max = np.max(np.abs(np.mean(np.array(good_subjects_shap),axis=0)))
    v_min = -v_max

    plot_topomap(np.mean(np.array(good_subjects_shap),axis=0),SENSORS,
                title = f"Subjects with a classif score $\\geq$ {SCORE_THRESHOLD}\n Mean Score : {np.mean(np.array(good_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f'{OUT_DIR}/Good_subjects.pdf',
                cbar_type='Permutation',
                suptitle = f"Feature Permutation Importance on {args.dataset}")


    plot_topomap(np.mean(np.array(bad_subjects_shap),axis=0),SENSORS,
                title = f"Subjects with a classif score $< {SCORE_THRESHOLD}$\n Mean Score : {np.mean(np.array(bad_subjects_scores)):.2f}",
                vlim=(v_min,v_max),
                savefile_name=f'{OUT_DIR}/Bad_subjects.pdf',
                cbar_type='Permutation',
                suptitle = f"Feature Permutation Importance on {args.dataset}")