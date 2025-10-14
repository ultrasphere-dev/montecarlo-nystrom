import jax
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace, to_device
from cyclopts import App
from matplotlib import pyplot as plt
from rich import print
from scipy.special import hankel1

from ._main import montecarlo_nystrom

app = App()


@app.command
def case(
    case_num: int,
    N: int = 2000,
    M: int = 100,
    n_plot: int = 2000,
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

        x = xp.linspace(0, 1, n_plot, device=device)[:, None]
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
                * -1j
                / (4 * xp.pi)
                * hankel1(0, k_ * xp.linalg.vector_norm(x - y, axis=-1))
                # * us.fundamental_solution(2, x - y, k_) # <- buggy
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

        def solve(A: Array, b: Array, /) -> Array:
            print("Using JAX CG solver")
            jax_device = jax.devices(device)[0]
            A, b = (
                jax.numpy.asarray(A, device=jax_device),
                jax.numpy.asarray(b, device=jax_device),
            )
            x = jax.numpy.stack(
                [jax.scipy.sparse.linalg.cg(A[i], b[i])[0] for i in range(A.shape[0])]
            )
            return xp.from_dlpack(x)

        x = xp.moveaxis(us.random_ball(c, shape=(n_plot,), xp=xp, device=device), 0, -1)
        zf = montecarlo_nystrom(
            random_samples=rho,
            kernel=kernel,
            rhs=rhs,
            n=N,
            n_mean=M,
            # solve=solve,
        )
        z = zf(x)
        x, z = to_device(x, "cpu"), to_device(z, "cpu")
        fig, ax = plt.subplots()
        sc = ax.scatter(x[:, 0], x[:, 1], c=xp.real(z), cmap="jet", vmin=-2, vmax=2)
        fig.colorbar(sc, ax=ax, label="Re z")
        ax.set_title(f"Case {case_num}, M={M}, N={N}, k={k}, m={m}")
        fig.savefig(f"case_{case_num}_m_{M}_n_{N}.png")
    if case_num == 5:
        import ultrasphere as us
        from scipy.special import spherical_jn, spherical_yn

        def harmonic_capacity_sphere(k: int, rho: int = 1) -> float:
            res = (
                -4
                * np.pi
                / (
                    1j
                    * k
                    * spherical_jn(0, k * rho)
                    * (spherical_jn(0, k * rho) + 1j * spherical_yn(0, k * rho))
                )
            )
            return float(res)

        k = 1
        s = 1
        cap = harmonic_capacity_sphere(k)
        print("Harmonic capacity of unit sphere:", cap)

        c = 1 + (1j * k * s) / (4 * np.pi) * cap

        def fundamental_solution_3d(x: Array, y: Array, k: int, /) -> Array:
            xp = array_namespace(x, y)
            r = xp.linalg.vector_norm(x - y, axis=-1)
            return xp.exp(1j * k * r) / (4 * xp.pi * r)

        def kernel(x: Array, y: Array, /) -> Array:
            return -cap * s * fundamental_solution_3d(x, y, k) / c

        def rhs(x: Array, /) -> Array:
            xp = array_namespace(x)
            return -cap * xp.exp(1j * k * x[..., 0]) / c

        def rho(n: int, /) -> Array:
            return xp.moveaxis(
                us.random_ball(us.create_standard(2), shape=(n,), xp=xp, device=device),
                0,
                -1,
            )

        zf = montecarlo_nystrom(
            random_samples=rho,
            kernel=kernel,
            rhs=rhs,
            n=N,
            n_mean=M,
        )
        x = xp.moveaxis(
            us.random_ball(us.create_polar(), shape=(n_plot,), xp=xp, device=device),
            0,
            -1,
        )
        x = xp.concat(
            [x, xp.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)], axis=-1
        )
        z = zf(x)
        u = -cap * z
        x, u = to_device(x, "cpu"), to_device(u, "cpu")
        fig, ax = plt.subplots()
        sc = ax.scatter(x[:, 0], x[:, 1], c=xp.real(u), cmap="jet", vmin=-2, vmax=2)
        fig.colorbar(sc, ax=ax, label="Re u")
        ax.set_title(f"Case {case_num}, M={M}, N={N}, k={k}, m={s}, cap={cap:.2f}")
        fig.savefig(f"case_{case_num}_m_{M}_n_{N}.png")
    else:
        raise ValueError(f"Invalid case number: {case_num}")
