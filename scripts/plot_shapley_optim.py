import numpy as np
from utils.Visualization import visualize_channel_importance

for i in range (1,10):

    shap_values = np.load(f"Results/Shapley/MDM/BNCI2014_001/Shapley_optim/shap_values_subject_{i}_mdm.npy")
    print(shap_values)
    print(np.max(shap_values))
    v_max = np.max(np.mean(shap_values,axis=0))
    v_min = -v_max


    sensors_bnci = [
            "Fz",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "P1",
            "Pz",
            "P2",
            "POz",
        ]

    sensors_dreyer = [
        'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2',
    'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'
    ]

    sensors = sensors_bnci

    visualize_channel_importance(
                importance_values=np.mean(shap_values,axis=0), 
                sensors=sensors,
                title=f"Subject 82",
                show_cbar=True,
                savefile_name= f"Results/Shapley/MDM/BNCI2014_001/Shapley_optim/shap_values_subject_{i}_mdm.pdf",
                cmap = 'PiYG',
                vlim=(v_min,v_max)
            )
            
  
