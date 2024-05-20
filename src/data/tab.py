from common.utils import minmax, reg_to_clf_target

import pandas as pd
import numpy as np

target_cols = ["Aphasia Type", "Severity"]
categoric_cols = ["Gender", "Stroke type", "Handedness"]
unusable_types = ["Semantic", "Acoustic-mnestic + Sensory", "Amnestic", "Other (neurodynamics)"]
usable_types = ["Efferent motor + Afferent Motor", "Sensory", "Efferent motor", "Dynamic", "Acoustic-mnestic", "Dysarthria", "Afferent motor"]
type_map = {
    "Efferent motor + Afferent Motor": 0,
    "Sensory": 1,
    "Efferent motor": 2,
    "Dynamic": 3,
    "Acoustic-mnestic": 4,
    "Dysarthria": 5,
    "Afferent motor": 6
}
severity_map = {
    "Very mild": 0,
    "Mild": 0,
    "Mild-moderate": 1,
    "Moderate": 2,
    "Moderate-severe": 3,
    "Severe": 4,
    "Very severe": 5,
    "Very-severe": 5
}

def get_tabular_data(matter, demo=False, method="clf", target="type", sev_from_asa=True, filter_acute=False, scale_reg_target=False, path="data"):
    wm = pd.read_excel(f"{path}/wm.xlsx").drop(columns=["N", "ID"]).reset_index().astype(float)
    gm = pd.read_excel(f"{path}/gm.xlsx").drop(columns=["N", "ID"]).reset_index().astype(float)
    demo_values = pd.read_excel(f"{path}/demo.xlsx").drop(columns=["N", "ID", "N of strokes"]).reset_index()
    asa = pd.read_excel(f"{path}/asa.xlsx").drop(columns=["N", "ID"]).reset_index()
    target_values = None

    wm.loc[:, "Lesion volume"] = minmax(wm["Lesion volume"]).astype(float)
    if matter == "grey":
        df = gm
        df.insert(1, "lesions", wm["Lesion volume"])
    elif matter == "white":
        df = wm
    else:
        df = pd.concat([gm, wm], axis=1)

    if demo:
        demo_values["Handedness"] = demo_values["Handedness"].fillna("left")
        demo_values.replace("relearnt left", "left", inplace=True)
        demo_values.replace("right/ambi", "ambi", inplace=True)
        demo_dummies = pd.get_dummies(demo_values.drop(columns=target_cols), columns=categoric_cols).astype(float)
        demo_dummies.loc[demo_dummies["Handedness_ambi"], ["Handedness_left", "Handedness_right"]] = 1
        demo_dummies.loc[demo_dummies["Stroke type_hemorrhagic + ischemic"], ["Stroke type_hemorrhagic", "Stroke type_ischemic"]] = 1
        demo_dummies.drop(columns=["Handedness_ambi", "Stroke type_hemorrhagic + ischemic", "Stroke type_aneurysm"], inplace=True)
        demo_dummies = minmax(demo_dummies)
        df = pd.concat([df, demo_dummies], axis=1)

    usable_mask = pd.Series([True] * df.shape[0])
    if filter_acute:
        usable_mask = demo_values["Post onset"].astype(int) >= 3

    if target == "type":
        usable_mask &= demo_values["Aphasia Type"].isin(usable_types) & demo_values["Aphasia Type"].notna()
        target_values = demo_values.loc[usable_mask, "Aphasia Type"].map(type_map).to_numpy(dtype=np.int32)
        n_classes = 7  # FIX!!!
    elif target == "severity" or target == "sev":
        usable_mask &= demo_values["Severity"].notna()
        target_values = demo_values.loc[usable_mask, "Severity"].map(severity_map)
        if sev_from_asa:
            target_values.loc[asa["ASA_before"].notna()] = asa.loc[asa["ASA_before"].notna(), "ASA_before"].apply(reg_to_clf_target, target="asa")
        target_values = target_values.to_numpy(dtype=np.int32 if method == "clf" else np.float32)
        n_classes = 6  # FIX!!!
    elif target == "asa":
        usable_mask &= asa["ASA_before"].notna()
        target_values = asa[usable_mask]["ASA_before"].to_numpy(dtype=np.float32)
        n_classes = 6  # FIX!!!
    else:
        raise ValueError
    
    X = df[usable_mask].drop(columns=["index"]).to_numpy(dtype=np.float32)
    if method == "reg" and scale_reg_target:
        target_values = minmax(target_values)

    return X, target_values, X.shape[1], n_classes
