import numpy as np

# Tolerances on the relative suboptimality, offered in the summary plots.
TOLERANCES = [1e-10, 1e-6]
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
    # The report shortens the names, and may merge solver variants or
    # datasets parameters: use the full names it keeps when available.
    name = {c: f"{c}_full_name" if f"{c}_full_name" in df else f"{c}_name"
            for c in ["dataset", "objective", "solver"]}
    problem = [name["dataset"], name["objective"]]
    f_star = df.groupby(problem)["objective_value"].transform("min")
    sub_opt = (df["objective_value"] - f_star) / f_star.abs()
    key = problem + [name["solver"]]
    reached = df[sub_opt <= tolerance].groupby(key + ["idx_rep"])["time"].min()
    n_reps = df.groupby(key)["idx_rep"].nunique()
    times = reached.groupby(key).median().reindex(n_reps.index)
    n_reached = reached.groupby(key).size().reindex(n_reps.index, fill_value=0)
    times[n_reached < n_reps / 2] = np.inf
    return times.unstack(name["solver"])


def performance_profile(times):
    """Fraction of the problems solved within each tau x the fastest time."""
    ratio = times.div(times.min(axis=1), axis=0).fillna(np.inf)
    return {
        solver: [(ratio[solver] <= tau).mean() for tau in TAUS]
        for solver in times.columns
    }
