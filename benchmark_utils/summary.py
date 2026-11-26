import numpy as np

# Tolerances on the relative suboptimality, offered in the summary plots.
TOLERANCES = [1e-10, 1e-6, 1e-4]
# Timeout of the runs in neurips2022_cases.yml, which the results do not store.
TIMEOUT = 600
# Grid of tau, the time relative to the fastest solver, for the profiles.
TAUS = np.logspace(0, 3.5, 100)


def time_to_tolerance(df, tolerance):
    """Time each solver takes to solve each problem, a (dataset, objective).

    A problem is solved when the relative suboptimality, with f* the best
    value of all solvers, is below ``tolerance``. The time is the median over
    the repetitions, inf when most repetitions do not reach the tolerance, and
    NaN when the solver did not run on the problem.

    Returns a DataFrame with one row per problem and one column per solver.
    """
    problem = ["dataset_name", "objective_name"]
    f_star = df.groupby(problem)["objective_value"].transform("min")
    sub_opt = (df["objective_value"] - f_star) / f_star.abs()
    key = problem + ["solver_name"]
    reached = df[sub_opt <= tolerance].groupby(key + ["idx_rep"])["time"].min()
    n_reps = df.groupby(key)["idx_rep"].nunique()
    times = reached.groupby(key).median().reindex(n_reps.index)
    n_reached = reached.groupby(key).size().reindex(n_reps.index, fill_value=0)
    times[n_reached < n_reps / 2] = np.inf
    return times.unstack("solver_name")


def performance_profile(times):
    """Fraction of the problems solved within each tau x the fastest time."""
    ratio = times.div(times.min(axis=1), axis=0).fillna(np.inf)
    return {
        solver: [(ratio[solver] <= tau).mean() for tau in TAUS]
        for solver in times.columns
    }
