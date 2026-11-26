from benchopt import BasePlot

import numpy as np

from benchmark_utils.summary import TIMEOUT, TOLERANCES, time_to_tolerance


def _reliability_speed(df, tolerance, unsolved):
    """Fraction of the problems each solver solves, and its median time
    relative to the most reliable solver.

    With ``unsolved="excluded"``, the median is over the problems both solve.
    With ``unsolved="timeout"``, an unsolved problem counts with the time
    ``TIMEOUT``, and the median is over all the problems the solver ran on.
    """
    times = time_to_tolerance(df, tolerance)
    solved = (times < np.inf).sum() / len(times)
    # The most reliable solver, the fastest one in case of ties.
    ref = min(times.columns, key=lambda s: (-solved[s], times[s].median()))
    if unsolved == "timeout":
        times = times.replace(np.inf, TIMEOUT)
    speed = {}
    for solver in times.columns:
        both = (times[solver] < np.inf) & (times[ref] < np.inf)
        if both.any():
            speed[solver] = np.median(times.loc[both, solver]
                                      / times.loc[both, ref])
    return solved, speed, ref


class Plot(BasePlot):
    name = "Pareto"
    type = "scatter"
    options = {"tolerance": TOLERANCES, "unsolved": ["excluded", "timeout"]}

    def plot(self, df, tolerance, unsolved):
        solved, speed, _ = _reliability_speed(df, tolerance, unsolved)
        traces = [
            dict(x=[speed[s]], y=[solved[s]], label=s, marker_size=16,
                 **self.get_style(s))
            for s in speed
        ]
        # The frontier links the solvers that no other is both faster and
        # more reliable than, as a staircase.
        x, y = [], []
        for s in sorted(speed, key=speed.get):
            if not y or solved[s] > y[-1]:
                x += [speed[s], speed[s]]
                y += [y[-1] if y else solved[s], solved[s]]
        traces.append(dict(x=x, y=y, label="Pareto frontier",
                           color=(0, 0, 0, .5), marker=0, marker_size=0))
        return traces

    def get_metadata(self, df, tolerance, unsolved):
        _, _, ref = _reliability_speed(df, tolerance, unsolved)
        where = ("unsolved problems at the timeout" if unsolved == "timeout"
                 else "on the problems both solve")
        return {
            "title": f"Problems solved to {tolerance:g} vs speed",
            "xlabel": f"Median time relative to {ref}, {where}",
            "ylabel": "Fraction of the problems solved",
        }
