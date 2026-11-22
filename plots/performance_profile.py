from benchopt import BasePlot

from benchmark_utils.summary import (
    TAUS, TOLERANCES, performance_profile, time_to_tolerance
)


class Plot(BasePlot):
    name = "Performance profile"
    type = "scatter"
    options = {"tolerance": TOLERANCES}

    def plot(self, df, tolerance):
        profile = performance_profile(time_to_tolerance(df, tolerance))
        return [
            dict(x=TAUS.tolist(), y=fraction, label=s, **self.get_style(s))
            for s, fraction in profile.items()
        ]

    def get_metadata(self, df, tolerance):
        return {
            "title": f"Performance profile, tolerance {tolerance:g}",
            "xlabel": "Time relative to the fastest solver on each problem",
            "ylabel": "Fraction of the problems solved",
        }
