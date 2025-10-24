import os

import matplotlib as mpl
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.transforms import ScaledTranslation
import numpy as np
import seaborn as sns
from NIE_tools import gather_QFIMs_eigvals_diff_noise, find_R_max, I_r_trend_truncated

sns.set(style="ticks", font_scale=1.5)

from jax.scipy.linalg import eigh

# setting up latex fontsyle
mpl.pyplot.rc("text", usetex=True)
mpl.pyplot.rcParams.update({"text.usetex": True})
mpl.pyplot.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": "Computer Modern"}
)


n_run = 10
epochs = 1000
absolute_path = os.path.abspath(__file__)
dir_path = os.path.dirname(absolute_path)  # '.' #

noise_standard = (
    np.array(
        [
            0,
            1e-5,
            1e-3,
            5e-3,
            1e-2,
            2e-2,
            3e-2,
            4e-2,
            5e-2,
            6e-2,
            7e-2,
            8e-2,
            9e-2,
            1e-1,
            2e-1,
            4e-1,
            7e-1,
        ]
    )
    * 1e-1
)  #


d = {}

datasets_list = [
    "sin",
    "sin2",
    "diabetes",
]

for design in ["2design", "paper_model"]:  #
    if design == "2design":
        layers_list = [4, 10]
        n_qubits = 5
        inner_layers = 3
    elif design == "paper_model":
        layers_list = [3, 5]
        n_qubits = 4
        inner_layers = 2
        noise_standard = np.concatenate(
            [noise_standard, [10**i for i in np.linspace(-3, 0, 13)]]
        )
        noise_standard = np.concatenate([noise_standard,np.array([3,5,8,9])*1e-2])

    params_per_layer = inner_layers * n_qubits

    for layers in layers_list:
        if design == "2design" and layers == 4:
            panel_label = "a)"
        elif design == "2design" and layers == 10:
            panel_label = "b)"
        elif design == "paper_model" and layers == 3:
            panel_label = "c)"
        elif design == "paper_model" and layers == 5:
            panel_label = "d)"

        fig1, ax1 = plt.subplots(
            1, int(len(datasets_list)), figsize=(5, 3), sharey=True
        )


        for j, noise_type in enumerate(["depolarizing", "phase_damp", "ampl_damp"]):
            if noise_type == "depolarizing":
                noise_lab = "DP"
            elif noise_type == "phase_damp":
                noise_lab = "PD"
            elif noise_type == "ampl_damp":
                noise_lab = "AD"

            histo_nid = []
            histo_mse = []
            histo_gap = []
            histo_labels = []

            for dataset in datasets_list:  #'uniform', 'random'
                if dataset == "sin":
                    data_lab = "S"
                elif dataset == "sin2":
                    data_lab = "S2"
                elif dataset == "diabetes":
                    data_lab = "D"

                ##### QFIM analysis
                data_folder_overp = (
                    dir_path + f"/data/NIE/{dataset}/{design}/{noise_type}/{layers}_layers/QFIMs"
                )


                # creating a dictionary gathering all noise levels
                gather_eigvs = gather_QFIMs_eigvals_diff_noise(noise_standard, dataset, data_folder_overp)

                ## we determin R_max,
                ## the index separating the eigenvalues that can grow from the one that are supppressed
                R_max = find_R_max(noise_standard, gather_eigvs)

                ## removing noiseless case
                noise_values_plotting = sorted(noise_standard)[1:]

                ## compute I_r only for the eigenvalues that can grow
                increases_list, mean_increase, std_increase = I_r_trend_truncated(
                    gather_eigvs, R_max, noise_values_plotting
                )

                ### increases_list has shape = (noise_values, R_max)

                ## computing p* inducing the best NIE
                opt_p_list_qfim = []
                for I_r_runs in np.array(
                    increases_list, dtype=object
                ).T:  # per each eigenval r we have multiple runs with different values of p
                    ## np.array(increases_list).T.shape = (eigenvals, inputs x runs, noise_levels)

                    tmp = list(
                        map(lambda x: noise_values_plotting[np.nanargmax(x)], I_r_runs)
                    )
                    opt_p_list_qfim.append(tmp)

                mean_p_qfim = np.mean(opt_p_list_qfim)
                std_p_qfim = np.std(opt_p_list_qfim)

                histo_nid.append((mean_p_qfim, std_p_qfim))
                

                #################################
                #### MSE
                data_folder = (
                    dir_path
                    + f"/data/regularization/{dataset}/{design}/{noise_type}/{layers}_layers"
                )

                train_history = np.load(
                    data_folder + f"/train_loss_{n_run}.npy", allow_pickle=True
                )
                test_history = np.load(
                    data_folder + f"/test_loss_{n_run}.npy", allow_pickle=True
                )
                opt_params = np.load(
                    data_folder + f"/opt_params_{n_run}.npy", allow_pickle=True
                )

                train_history = train_history[()]
                test_history = test_history[()]
                opt_params = opt_params[()]

                ## possible reduction of noise levels plotted
                selected = sorted(noise_standard)  

                avg_train = []
                avg_test = []

                error_train = []
                error_test = []

                train_mse_list = []
                test_mse_list = []

                for k in selected:
                    train_losses = np.array(train_history[k])
                    test_losses = np.array(test_history[k])

                    train_losses = train_losses[:, :epochs]
                    test_losses = test_losses[:, :epochs]
                    ## train

                    final_train_losses = list(map(lambda x: x[-1], train_losses))
                    train_mse_list.append(final_train_losses)

                    mean_train_history = np.mean(train_losses, axis=0)[:epochs]
                    std_train_history = np.std(
                        train_losses,
                        axis=0,
                    )[:epochs]

                    mean_train_history = mean_train_history.reshape((epochs,))
                    std_train_history = std_train_history.reshape((epochs,))

                    avg_train.append(mean_train_history[epochs - 1])
                    error_train.append(std_train_history[epochs - 1])

                    ## test

                    final_test_losses = list(map(lambda x: x[-1], test_losses))
                    test_mse_list.append(final_test_losses)
                    # print(test_losses[:,-1])
                    mean_test_history = np.mean(test_losses, axis=0)
                    std_test_history = np.std(
                        test_losses,
                        axis=0,
                    )

                    mean_test_history = mean_test_history.reshape((epochs,))
                    std_test_history = std_test_history.reshape((epochs,))

                    avg_test.append(mean_test_history[epochs - 1])
                    error_test.append(std_test_history[epochs - 1])

                ## determining opt p per each run
                ## p* can be gen gap=0 or dip in test error

                train_mse_list = np.array(train_mse_list)
                test_mse_list = np.array(test_mse_list)

                list_gen_gaps = np.abs(train_mse_list - test_mse_list)
                list_argmin_gen_gap = list(
                    map(lambda x: selected[np.nanargmin(x)], list_gen_gaps.T)
                )

                list_argmin_test_mse = list(
                    map(lambda x: selected[np.nanargmin(x)], test_mse_list.T)
                )
                mean_p_gap = np.mean(list_argmin_gen_gap)
                std_p_gap = np.std(list_argmin_gen_gap)

                mean_p_test_mse = np.mean(list_argmin_test_mse)
                std_p_test_mse = np.std(list_argmin_test_mse)
                
                histo_mse.append((mean_p_test_mse, std_p_test_mse))
                histo_gap.append((mean_p_gap, std_p_gap))
                histo_labels.append(f"{data_lab}")

                list_mse = list_argmin_test_mse

                


            ## bars
            ax1[j].set_title(noise_lab)
            ax1[j].bar(
                np.arange(len(histo_nid)) - 0.25,
                np.array(histo_nid)[:, 0],
                yerr=np.array(histo_nid)[:, 1],
                # fmt='D',
                capsize=5,
                label="NIE",
                width=0.25,
            )
            ax1[j].bar(
                np.arange(len(histo_nid)),
                np.array(histo_mse)[:, 0],
                yerr=np.array(histo_mse)[:, 1],
                # fmt='o',
                capsize=5,
                label="MSE",
                width=0.25,
            )
            ax1[j].bar(
                np.arange(len(histo_nid)) + 0.25,
                np.array(histo_gap)[:, 0],
                yerr=np.array(histo_gap)[:, 1],
                # fmt='s',
                capsize=5,
                label="Gen. gap",
                width=0.25,
            )
            ax1[j].set_yscale("log")
            if design == "2design":
                ax1[j].set_ylim(5e-4, 1e-1)
            elif design == "paper_model":
                ax1[j].set_ylim(1e-3, 1)
            ax1[j].set_xlim(-0.5, 2.5)
            ax1[j].set_xticks(range(len(histo_nid)))
            ax1[j].set_xticklabels(histo_labels)
            # d[f'{design}+{dataset}+{noise_type}+{layers}+mse'] = [mean_p_mse,std_p_mse]

        ax1[0].set_ylabel(r"$p^*$")
        lgd = ax1[1].legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=3, columnspacing=0.6
        )

        ax1[0].text(
            -0.85,
            1.09,
            panel_label,
            transform=(
                ax1[0].transAxes
                + ScaledTranslation(+10 / 72, -20 / 72, fig1.dpi_scale_trans)
            ),
            weight="bold",
            # fontsize='medium',
            va="bottom",
            # fontfamily='serif'
        )


        fig1.subplots_adjust(hspace=0.3, top=0.65, bottom=0.2, left=0.2, right=0.95)
        # fig1.savefig(
        #     dir_path + f"/plots/regularization/compare_opt_p_{design}_{layers}.pdf",
        #     bbox_extra_artists=(lgd,),
        #     bbox_inches="tight",
        # )
plt.show()
