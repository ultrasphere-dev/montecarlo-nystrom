import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace, to_device
from cyclopts import App
from matplotlib import pyplot as plt

from ._main import montecarlo_nystrom

app = App()


@app.command
def case(
    case_num: int,
    N: int = 2000,
    M: int = 100,
    backend: str = "torch",
    device: str | None = None,
) -> None:
    """Compute examples in the paper."""
    if backend == "numpy":
        from array_api_compat import numpy as xp
    elif backend == "torch":
        from array_api_compat import torch as xp
    if device is None:
        if backend == "torch":
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = "cpu"

    if 1 <= case_num <= 3:

        def kernel(x: Array, y: Array, /) -> Array:
            xp = array_namespace(x, y)
            alpha = xp.asarray(0.4)  # L472
            return xp.linalg.vector_norm(x - y, axis=-1) ** (-alpha)

        def rhs(x: Array, /) -> Array:
            xp = array_namespace(x)
            x = x[..., 0]
            if case_num == 1:
                return xp.ones_like(x)
            elif case_num == 2:
                return (1 - x) * x
            elif case_num == 3:
                return xp.sin(6 * xp.pi * x)
            raise NotImplementedError()

        def p(i: int, /) -> Array:
            rng = np.random.default_rng(0)
            return xp.asarray(rng.uniform(0, 1, size=(i, 1)), device=device)

        x = xp.linspace(0, 1, N, device=device)[:, None]
        zf = montecarlo_nystrom(
            random_samples=p,
            kernel=kernel,
            rhs=rhs,
            n=N,
            n_mean=M,
        )
        z = zf(x)
        fig, ax = plt.subplots()
        x, z = to_device(x, "cpu"), to_device(z, "cpu")
        ax.plot(x[:, 0], z, label="Approximate solution")
        ax.set_title(f"Case {case_num}, M={M}, N={N}")
        ax.legend()
        fig.savefig(f"case_{case_num}_m_{M}_n_{N}.png")
    if case_num == 4:
        import ultrasphere as us

        c = us.create_polar()
        k = 5
        m = 10

        def kernel(x: Array, y: Array, /) -> Array:
            xp = array_namespace(x, y)
            k_ = xp.asarray(k)
            m_ = xp.asarray(m)
            return (
                c.volume()
                * (m_ - 1)
                * k_**2
                * us.fundamental_solution(xp.asarray(2), x - y, k_)
            )

        def rhs(x: Array, /) -> Array:
            xp = array_namespace(x)
            return xp.exp(xp.asarray(1j * k) * x[..., 0])

        def rho(n: int, /) -> Array:
            return xp.moveaxis(
                us.random_ball(c, shape=(n,), xp=xp, device=device),
                0,
                -1,
            )

        x = xp.moveaxis(us.random_ball(c, shape=(N,), xp=xp, device=device), 0, -1)
        zf = montecarlo_nystrom(
            random_samples=rho,
            kernel=kernel,
            rhs=rhs,
            n=N,
            n_mean=M,
        )
        z = zf(x)
        x, z = to_device(x, "cpu"), to_device(z, "cpu")
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], c=xp.real(z), cmap="jet")
        fig.savefig(f"case_{case_num}_m_{M}_n_{N}.png")
