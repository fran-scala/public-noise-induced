import numpy as np
from jax.scipy.linalg import eigh


def gather_QFIMs_eigvals_diff_noise(noise_values, dataset, data_folder_overp):
    """Creating a dictionary averaging over the training dataset"""

    gather_eigvs = {}

    for p in noise_values:
        if dataset == "diabetes2":
            qfims = np.load(
                data_folder_overp + f"/{dataset[:-1]}_QFIMs_noisy_{p}.npy",
                allow_pickle=True,
            )[()][p]

        else:
            qfims = np.load(
                data_folder_overp + f"/{dataset}_QFIMs_noisy_{p}.npy", allow_pickle=True
            )[()][p]

        ## keepeing only the eigenvals
        w_s = list(
            map(
                lambda x: np.array(
                    eigh(np.array(x, dtype=np.float64))[0], dtype=np.float64
                ),
                qfims,
            )
        )
        ## sorting and setting all positive
        ### (the only negatives are ~0 and due to numerical issues)
        ### the QFIM is positive semidefinite
        w_s = np.sort(np.abs(w_s))

        gather_eigvs[p] = np.array(w_s, dtype=np.float64)

    return gather_eigvs


def find_R_max(noise_values, gather_eigvs):
    """## Determin R_max, the index separating the eigenvalues that can grow from the one that are supppressed"""

    R_list = []
    for i, p in enumerate(sorted(noise_values)):
        ## in case there is an eigenvalue that is exactly zero we substitute it with 1e-10 to avoid numerical issues
        noiseless_eigvs = np.where(
            np.array(gather_eigvs[0.0], dtype=np.float64) == 0.0,
            1e-10,
            gather_eigvs[0.0],
        )

        increase = np.array(gather_eigvs[p], dtype=np.float64) / noiseless_eigvs
        rows, cols = np.where(
            np.array(increase) > 1
        )  # indexes with I_r > 1 --> for matrices
        R_tmp = np.zeros(increase.shape[0])  # array with default threshold 0
        np.maximum.at(R_tmp, rows, cols)

        R_list.append(R_tmp)
    R_array = np.vstack(R_list)
    R_max_list = np.max(R_array, axis=0)  ### take the highest index as a cutoff
    R_max_list = R_max_list.astype(int)
    R_max = int(round(np.min(R_max_list))) + 1
    return R_max


def I_r_trend_truncated(gather_eigvs, R_max, noise_values_plotting):
    """Compute I_r only for the eigenvalues that can grow"""

    increases_list = []  # shape = (noise_values, R_max)
    mean_increase = []
    std_increase = []
    for i, p in enumerate(noise_values_plotting):
        ## in case there is an eigenvalue that is exactly zero we substitute it with 1e-10 to avoid numerical issues
        noiseless_eigvs = np.where(
            np.array(gather_eigvs[0.0], dtype=np.float64) == 0.0,
            1e-10,
            gather_eigvs[0.0],
        )
        increase = np.array(gather_eigvs[p], dtype=np.float64) / noiseless_eigvs

        ## we truncate at R_max
        truncated_incr = [increase[k][:R_max] for k in range(increase.shape[0])]
        increases_list.append(truncated_incr)  #
        mean_increase.append(np.mean(np.concatenate(truncated_incr)))
        std_increase.append(np.std(np.concatenate(truncated_incr)))

    ## quantities that we plot
    mean_increase = np.array(mean_increase)
    std_increase = np.array(std_increase)

    return increases_list, mean_increase, std_increase
