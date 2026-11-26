from benchopt import BasePlot

import numpy as np

from benchmark_utils.summary import (
    TAUS, TOLERANCES, performance_profile, time_to_tolerance
)


class Plot(BasePlot):
    """Area under the performance profile, over log(tau), normalized.

    It sums up speed and reliability in one number: 1 when a solver is the
    fastest on all the problems, 0 when it solves none.
    """
    name = "Profile area"
    type = "bar_chart"
    options = {"tolerance": TOLERANCES}

    def plot(self, df, tolerance):
        profile = performance_profile(time_to_tolerance(df, tolerance))
        log_taus = np.log10(TAUS)
        area = {
            s: np.trapezoid(fraction, log_taus) / log_taus[-1]
            for s, fraction in profile.items()
        }
        return [
            dict(y=[a], label=s, text=f"{a:.2f}",
                 color=self.get_style(s)["color"])
            for s, a in sorted(area.items(), key=lambda item: -item[1])
        ]

    def get_metadata(self, df, tolerance):
        return {
            "title": f"Area under the performance profile, tolerance "
                     f"{tolerance:g}",
            "ylabel": "Normalized area (1: fastest on all the problems)",
        }
