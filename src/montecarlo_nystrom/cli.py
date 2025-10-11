import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from cyclopts import App
from matplotlib import pyplot as plt

from ._main import montecarlo_nystrom

app = App()


@app.command
def case(case_num: int, N: int = 2000) -> None:
    """Compute examples in the paper."""
    if 1 <= case_num <= 3:

        def kernel(x: Array, y: Array, /) -> Array:
            xp = array_namespace(x, y)
            alpha = xp.asarray(0.4)  # L472
            return xp.linalg.vector_norm(x - y, axis=-1) ** (-alpha)

        def rhs(x: Array, /) -> Array:
            xp = array_namespace(x)
            if case_num == 1:
                return xp.ones_like(x)
            elif case_num == 2:
                return (1 - x) * x
            elif case_num == 3:
                return xp.sin(6 * xp.pi * x)
            raise NotImplementedError()

        def p(i: int, /) -> Array:
            rng = np.random.default_rng(0)
            return rng.uniform(0, 1, size=(i, 1))

        x = np.linspace(0, 1, N)[:, None]
        zf = montecarlo_nystrom(
            random_samples=p,
            kernel=kernel,
            rhs=rhs,
            n=N,
        )
        z = zf(x)
        fig, ax = plt.subplots()
        ax.plot(x[:, 0], z[:, 0], label="Approximate solution")
        ax.set_title(f"Case {case_num}")
        ax.legend()
        fig.savefig(f"case_{case_num}.png")
