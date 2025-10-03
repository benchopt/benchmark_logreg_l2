import re
import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


usetex = mpl.checkdep_usetex(True)
params = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.usetex": usetex,
}
mpl.rcParams.update(params)


# matplotlib style config
titlesize = 22
ticksize = 16
labelsize = 20

MARKERS = list(plt.Line2D.markers.keys())[:-4]
CMAP = plt.get_cmap('tab20')
#####

regex = re.compile(r'\[(.*?)\]')


SAVEFIG = True
PLOT_COLUMN = "objective_Train loss"
fignames = ["logreg_l2_final", "logreg_l2_supp_mat_lmbd"]

BENCH_FILES = [
    './outputs/logreg-l2_neurips.csv',
    './outputs/criteo_neurips.csv',
]
FLOATING_PRECISION = 1e-8
MIN_XLIM = 1e-3

GPU_SOLVERS = ['snapml[gpu=True]', 'cuml']


DICT_XLIM = {
    "rcv1[scaled=False]": 1e-2,
    "rcv1[scaled=True]": 1e-2,
    "news20[scaled=False]": 1e-1,
    "news20[scaled=True]": 1e-1,
    "ijcnn1[scaled=False]": 5e-3,
    "ijcnn1[scaled=True]": 5e-3,
    "criteo[scaled=False]": 5e-1,
    "criteo[scaled=True]": 5e-1,
}

YTICKS = (1e4, 1, 1e-4, 1e-8)

IDX_ROWS = [{
    ("ijcnn1", "lmbd=1]"): (0, 'ijcnn1'),
    ("madelon", "lmbd=1]"): (1, "madelon"),
    ("news20", "lmbd=1]"): (2, "news20.binary"),
    ("criteo", "lmbd=1]"): (3, "criteo")
}, {
    ("ijcnn1", "fit_intercept=False"): (0, 'ijcnn1'),
    ("madelon", "fit_intercept=False"): (1, "madelon"),
    ("news20", "fit_intercept=False"): (2, "news20.binary"),
    ("rcv1", "fit_intercept=False"): (3, "rcv1.binary"),
    ("adult", "fit_intercept=False"): (4, "adult"),
}
]
IDX_COLUMNS = [
    {
        ('scaled=False', "intercept=False"): (0, "Raw"),
        ('scaled=True', "intercept=False"): (1, "Scaled"),
        ('scaled=False', "intercept=True"): (2, "Intercept"),
    }, {
        ('scaled=False', "lmbd=0.1]"): (0, r"$\lambda=0.1$"),
        ('scaled=False', "lmbd=1]"): (1, r"$\lambda=1$"),
        ('scaled=False', "lmbd=10]"): (2, r"$\lambda=10$"),
    },
]

all_solvers = {
    'sklearn[liblinear]': 'scikit-learn[liblinear]',
    'snapml[gpu=False]': 'snapML[cpu]',
    'Lightning[method=sag]': 'lightning[sag]',
    'sklearn[sgd]': 'scikit-learn[sgd]',
    'sklearn[lbfgs]': 'scikit-learn[lbfgs]',
    'snapml[gpu=True]': 'snapML[gpu]',
    'Lightning[method=saga]': 'lightning[saga]',
    'sklearn[sag]': 'scikit-learn[sag]',
    'sklearn[newton-cg]': 'scikit-learn[newton-cg]',
    'Lightning[method=cd]': 'lightning[cd]',
    'svrg-tick': 'tick[svrg]',
    'sklearn[saga]': 'scikit-learn[saga]',
    'cuml': 'cuML[gpu]',
}

df = pd.read_csv(BENCH_FILES[0], header=0, index_col=0)
STYLE = {solver_name: (CMAP(i), MARKERS[i], all_solvers[solver_name])
         for i, solver_name in enumerate(df["solver_name"].unique())}

fontsize = 20
labelsize = 20


def filter_data_and_obj(dataset, objective, idx):
    for (p_data, p_obj), res in idx.items():
        if ((p_data is None or p_data in dataset)
                and (p_obj is None or p_obj in objective)):
            return res
    return None, None


for figname, idx_rows, idx_cols in zip(fignames, IDX_ROWS, IDX_COLUMNS):

    plt.close('all')

    n_rows, n_cols = len(idx_rows), len(idx_cols)
    main_fig, axarr = plt.subplots(
        n_rows,
        n_cols,
        sharex='row',
        sharey='row',
        figsize=[11, 1 + 2 * n_rows],
        constrained_layout=True, squeeze=False
    )

    for bench_file in BENCH_FILES:
        df = pd.read_csv(bench_file, header=0, index_col=0)
        datasets = df["data_name"].unique()
        objectives = df["objective_name"].unique()
        solvers = df["solver_name"].unique()
        solvers = np.array(sorted(solvers, key=str.lower))
        for dataset in datasets:
            for objective in objectives:
                idx_col, clabel = filter_data_and_obj(
                    dataset, objective, idx_cols
                )
                idx_row, rlabel = filter_data_and_obj(
                    dataset, objective, idx_rows
                )
                if None in [idx_row, idx_col]:
                    continue
                df2 = df.query(
                    'data_name == @dataset & objective_name == @objective'
                )
                ax = axarr[idx_row, idx_col]
                print(idx_row, idx_col, dataset, objective)

                c_star = np.min(df2[PLOT_COLUMN]) - FLOATING_PRECISION
                for i, solver_name in enumerate(all_solvers):

                    # Get style if it exists or create a new one
                    # idx_next = len(STYLE)
                    color, marker, label = STYLE.get(solver_name)
                    # STYLE[solver_name] = (color, marker, label)

                    df3 = df2.query('solver_name == @solver_name')
                    curve = df3.groupby('stop_val').median()

                    q1 = df3.groupby('stop_val')['time'].quantile(.1)
                    q9 = df3.groupby('stop_val')['time'].quantile(.9)
                    y = curve[PLOT_COLUMN] - c_star

                    ls = "--" if solver_name in GPU_SOLVERS else None
                    ax.loglog(
                        curve["time"], y, color=color, marker=marker,
                        label=label, linewidth=2, markevery=3, ls=ls,
                        markersize=6,
                    )

                ax.set_xlim(DICT_XLIM.get(dataset, MIN_XLIM), ax.get_xlim()[1])

                x1, x2 = ax.get_xlim()
                x1, x2 = np.ceil(np.log10(x1)), np.floor(np.log10(x2))

                y1, y2 = ax.get_ylim()
                ax.set_ylim(y1, 1e5 if 'criteo' not in dataset else 1e8)

                xticks = 10 ** np.arange(x1, x2+1)
                ax.set_xticks(xticks)
                axarr[idx_row, 0].set_yticks(YTICKS)

                axarr[0, idx_col].set_title(clabel, fontsize=labelsize)
                axarr[n_rows-1, idx_col].set_xlabel(
                    "Time (s)", fontsize=labelsize
                )

                ax.tick_params(axis='both', which='major', labelsize=ticksize)
                ax.grid()

                axarr[idx_row, 0].set_ylabel(rlabel, fontsize=labelsize)

        # main_fig.suptitle(regex.sub('', objective), fontsize=fontsize)
    plt.show(block=False)

    # plot legend on separate fig
    leg_fig, ax2 = plt.subplots(1, 1, figsize=(20, 4))
    n_col = 4
    if n_col is None:
        n_col = len(axarr[0, 0].lines)

    # take first ax, more likely to have all solvers converging
    ax = axarr[0, 0]
    lines_ordered = list(itertools.chain(
        *[ax.lines[i::n_col] for i in range(n_col)])
    )
    legend = ax2.legend(
        lines_ordered, [line.get_label() for line in lines_ordered],
        ncol=n_col, loc="upper center"
    )
    leg_fig.canvas.draw()
    leg_fig.tight_layout()
    width = legend.get_window_extent().width
    height = legend.get_window_extent().height
    leg_fig.set_size_inches((width / 80,  max(height / 80, 0.5)))
    plt.axis('off')
    plt.show(block=False)

    if SAVEFIG:
        Path('./figures').mkdir(exist_ok=True)
        main_fig_name = f"figures/{figname}.pdf"
        main_fig.savefig(main_fig_name, dpi=300)
        os.system(f"pdfcrop '{main_fig_name}' '{main_fig_name}'")
        main_fig.savefig(f"figures/{figname}.svg")

        leg_fig_name = f"figures/{figname}_legend.pdf"
        leg_fig.savefig(leg_fig_name, dpi=300)
        os.system(f"pdfcrop '{leg_fig_name}' '{leg_fig_name}'")
        leg_fig.savefig(f"figures/{figname}_legend.svg", dpi=300)
