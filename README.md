# IOH experiment examples

This guide shows how to create and log benchmark problems with [IOHexperimenter](https://iohprofiler.github.io/IOHexperimenter/): first a BBOB problem, then a practical problem from [BayesOptApp/Practical_Problems](https://github.com/BayesOptApp/Practical_Problems).

## Installation

Use a virtual environment to keep the experiment dependencies isolated:

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\activate` on Windows), then install the packages:

```bash
python -m pip install --upgrade pip
```

For the BBOB example alone:

```bash
python -m pip install ioh
```

To use the practical-problem example, install the repository:

```bash
python -m pip install "git+https://github.com/BayesOptApp/Practical_Problems.git"
```

Installing `Practical_Problems` also installs IOH and its required dependencies. It currently pins `ioh<=0.3.22` and `FLORIS==2.5.2`. If you want to run both examples in one environment, use the practical-problem installation so the compatible IOH version is installed. The `Bench_Fun_Eps` and `Bench_Fun_Pitz` problems additionally require Docker or Podman; the example below does not.

## BBOB problem

`ioh.get_problem` returns a problem instance. Choose a BBOB function ID, instance, and dimension, then configure an `Analyzer` and attach it to the problem:

```python
import os

import ioh
import numpy as np

problem = ioh.get_problem(
    1,                              # BBOB function ID
    instance=1,
    dimension=5,
    problem_class=ioh.ProblemClass.BBOB,
)

logger = ioh.logger.Analyzer(
    triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
    root=os.path.curdir,
    algorithm_name="RandomSearch",
    folder_name="Results-BBOB",
    algorithm_info="Uniform random search",
    store_positions=False,
)
problem.attach_logger(logger)

rng = np.random.default_rng(42)
best = float("inf")
try:
    for _ in range(1000):
        x = rng.uniform(-5, 5, size=problem.meta_data.n_variables)
        best = min(best, problem(x))
finally:
    problem.detach_logger()

print(f"Best objective: {best}")
```

The BBOB search domain is `[-5, 5]^d`. The logger writes experiment data under `Results-BBOB/`.

## Practical problem

`Practical_Problems` exposes its problems as IOH problems, so the logger setup and attachment work the same way. This example uses problem ID `1126` (`Spacecraft_Trajectory_OptimizationC1`), which has 26 variables:

```python
import os

import ioh
import numpy as np
from Practical_Problems import get_practical_problem

problem = get_practical_problem(1126, dim=26)

logger = ioh.logger.Analyzer(
    triggers=[ioh.logger.trigger.OnImprovement(), ioh.logger.trigger.Each(10)],
    root=os.path.curdir,
    algorithm_name="RandomSearch",
    folder_name="Results-Practical",
    algorithm_info="Uniform random search",
    store_positions=False,
)
problem.attach_logger(logger)

rng = np.random.default_rng(42)
best = float("inf")
try:
    for _ in range(1000):
        x = rng.uniform(-5, 5, size=problem.meta_data.n_variables)
        best = min(best, problem(x))
finally:
    problem.detach_logger()

print(f"Best objective: {best}")
```

Practical problems are presented to the optimizer in `[-5, 5]^d`; the package maps these values to each problem's internal bounds. The logger writes experiment data under `Results-Practical/`. Select a valid dimension for other problems from the [Practical_Problems README](https://github.com/BayesOptApp/Practical_Problems#available-problems).

Upload the generated IOH data to [IOHanalyzer](https://iohanalyzer.liacs.nl/) to inspect and compare experiment results.
