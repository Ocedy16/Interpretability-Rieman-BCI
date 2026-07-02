import argparse
from src.data.dataset_config import DATASET_CONFIG

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

    OUT_DIR = f'Results/Permutation/SPDNet/{arg.dataset}/SansNormalisation'

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

    results_dic = fit(X, y_encoded, meta, model_config, n_iter=args.n_splits, n_perm=args.n_perms)

    filename = os.path.join(OUT_DIR, 'results_dic_norm.pkl')
    with open(filename,'wb') as f:
        pickle.dump(results_dic,f)