from benchopt import BasePlot

import numpy as np

from benchmark_utils.summary import TOLERANCES, time_to_tolerance


def _reliability_speed(df, tolerance):
    """Fraction of the problems each solver solves, and its median time
    relative to the most reliable solver, on the problems both solve."""
    times = time_to_tolerance(df, tolerance)
    solved = (times < np.inf).sum() / len(times)
    # The most reliable solver, the fastest one in case of ties.
    ref = min(times.columns, key=lambda s: (-solved[s], times[s].median()))
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
    options = {"tolerance": TOLERANCES}

    def plot(self, df, tolerance):
        solved, speed, _ = _reliability_speed(df, tolerance)
        traces = [
            dict(x=[speed[s]], y=[solved[s]], label=s, **self.get_style(s),
                 description=f"{s}: {solved[s]:.0%} of the problems solved, "
                             f"{speed[s]:.2g}x the time of the reference")
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
                           color=(0, 0, 0, .5), marker=0))
        return traces

    def get_metadata(self, df, tolerance):
        _, _, ref = _reliability_speed(df, tolerance)
        return {
            "title": f"Problems solved to {tolerance:g} vs speed",
            "xlabel": f"Median time relative to {ref}",
            "ylabel": "Fraction of the problems solved",
        }
