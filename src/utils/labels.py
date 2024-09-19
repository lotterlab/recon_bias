import torch

sex_map = {"M": 0, "F": 1}
diagnosis_map = {
    "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted": 0,
    "Astrocytoma, IDH-wildtype": 1,
    "Astrocytoma, IDH-mutant": 2,
    "Glioblastoma, IDH-wildtype": 3,
}

def extract_labels_from_row(row):
    sex = torch.tensor(sex_map[row["Sex"]], dtype=torch.int64)
    age = torch.tensor(row["Age at MRI"], dtype=torch.float64)
    cns = torch.tensor(row["WHO CNS Grade"], dtype=torch.int64)
    diagnosis = torch.tensor(
        diagnosis_map[row["Final pathologic diagnosis (WHO 2021)"]],
        dtype=torch.int64,
    )
    censor = torch.tensor(-1 * (row["1-dead 0-alive"] - 1), dtype=torch.int64)
    os = torch.tensor(row["OS"], dtype=torch.float64)

    labels = torch.stack([sex, age, cns, diagnosis, censor, os])

    return labels