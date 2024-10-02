import numpy as np


def bootstrap(gt, pred, recon, f):
    observed_diff = f(gt, pred) - f(gt, recon)

    n = len(gt)
    boot_diffs = []

    for _ in range(1000):
        # Create a bootstrap sample with replacement
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        y_true_boot = np.array(gt)[indices]
        y_pred1_boot = np.array(pred)[indices]
        y_pred2_boot = np.array(recon)[indices]

        # Calculate C-index difference for the bootstrap sample
        boot_diff = f(y_true_boot, y_pred1_boot) - f(y_true_boot, y_pred2_boot)
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    # Calculate the p-value (two-sided test)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))

    return p_value
