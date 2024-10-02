import torch
import pandas as pd

sex_map = {"M": 0, "F": 1}
diagnosis_map = {
    "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted": 0,
    "Astrocytoma, IDH-wildtype": 1,
    "Astrocytoma, IDH-mutant": 2,
    "Glioblastoma, IDH-wildtype": 3,
}

def extract_labels_from_row(row, age_bins):
    sex = torch.tensor(sex_map[row["sex"]], dtype=torch.int64)
    age = torch.tensor(row["age_at_mri"], dtype=torch.float64)
    cns = torch.tensor(row["who_cns_grade"], dtype=torch.int64)
    diagnosis = torch.tensor(
        diagnosis_map[row["final_diagnosis"]],
        dtype=torch.int64,
    )
    censor = torch.tensor(-1 * (row["alive"] - 1), dtype=torch.int64)
    os = torch.tensor(row["os"], dtype=torch.float64)
    age_labels = list(range(0, len(age_bins) - 1))
    age_bucket = torch.tensor(pd.cut([row["age_at_mri"]], bins=age_bins, labels=age_labels, right=False))
    labels = torch.stack([sex, age, cns, diagnosis, censor, os, age_bucket[0]])

    return labels