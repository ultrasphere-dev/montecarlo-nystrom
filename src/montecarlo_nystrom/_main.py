from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace


def montecarlo_nystrom(
    *,
    random_samples: Callable[[int], Array],
    kernel: Callable[[Array, Array], Array],
    rhs: Callable[[Array], Array],
    n: int,
    n_mean: int = 1,
    solve: Callable[[Array, Array], Array] | None = None,
) -> Callable[[Array], Array]:
    r"""
    Solve integral equations of the second kind of the following form.

    $\forall d \in \mathbb{N}.$
    $\forall \Omega \in \mathbb{R}^d [\Omega \text{ is bounded Lipschitz}].$
    $\forall p \in L^\infty(\Omega, {\mathbb{R}}_{\geq 0}) [\int_\Omega p(y) dy = 1].$
    $\forall f, z \in L^2(\Omega,\mathbb{C}).$
    $\forall k \in L^2(\Omega,L^2(\Omega,\mathbb{C}))$
    $[z(y) + \int_\Omega k(y, y') z(y') p(y') dy' = f(y)].$

    Let $N \in \mathbb{N}$, $(y_i)_{i=1}^N$ be i.i.d. samples drawn from $p$.

    Let $(z_{N,i})_{i=1}^N$ be the solution of the linear system

    $$
    z_{N,i} + \frac{1}{N} \sum_{j=1}^N k(y_i, y_j) z_{N,j} = f(y_i)
    \quad i \in \{1, \ldots, N\}
    $$

    and

    $$
    z_N(y) := f(y) - \frac{1}{N} \sum_{i=1}^N k(y, y_i) z_{N,i}
    \quad y \in \Omega
    $$

    Then $z_N$ would approximate $z$ as $N \to \infty$.

    Parameters
    ----------
    random_samples : Callable[[int], Array]
        A function that takes an integer n and
        returns n i.i.d. samples from the distribution p of shape (..., n, d).
    kernel : Callable[[Array, Array], Array]
        Kernel function k of (..., :, :, d), (..., :, :, d) -> (..., :, :).

        The diagonal part will be automatically dropped
        and can be of any value.
    rhs : Callable[[Array], Array]
        Right-hand side function f of (..., :, d) -> (..., :).
    n : int
        The number of random samples to draw from the distribution p.
    n_mean : int
        The number of independent runs to average over, by default 1.
    solve : Callable[[Array, Array], Array] | None, optional
        A function that takes a matrix A of shape (..., n, n)
        and a right-hand side b of shape (..., n, 1)
        and returns the solution of shape (..., n, 1).
        If None, uses `xp.linalg.solve`, by default None.

    Returns
    -------
    Callable[[Array], Array]
        Approximate solution function z_N that takes an array of shape (...(x), ..., d)
        and returns an array of shape (...(x), ...).

    Examples
    --------
    >>> import numpy as np
    >>> from montecarlo_nystrom import montecarlo_nystrom
    >>> rng = np.random.default_rng(0)
    >>> def random_samples(n):
    ...     return rng.uniform(0, 1, size=(n, 1))
    >>> def kernel(x, y):
    ...     return np.linalg.vector_norm(x - y, axis=-1) ** -0.4
    >>> def rhs(x):
    ...     x0 = x[..., 0]
    ...     return np.ones_like(x0)
    >>> z_N = montecarlo_nystrom(
    ...     random_samples=random_samples,
    ...     kernel=kernel,
    ...     rhs=rhs,
    ...     n=100,
    ...     n_mean=10,
    ... )
    >>> np.round(z_N(np.asarray((0.5,))), 6)  # Evaluate at x=0.5
    np.float64(0.272957)

    """
    y = random_samples(n * n_mean)
    if y.shape[-2] != n * n_mean:
        raise ValueError(
            f"random_samples returned array of shape {y.shape}, "
            f"expected (..., {n * n_mean}, d)"
        )
    xp = array_namespace(y)
    y = xp.reshape(y, (*y.shape[:-2], n_mean, n, y.shape[-1]))  # (..., n_mean, n, d)
    K = kernel(y[..., :, :, None, :], y[..., :, None, :, :])  # (..., n_mean, n, n)
    b = rhs(y)  # (..., n_mean, n)
    if K.shape[-3:] != (n_mean, n, n):
        raise ValueError(
            f"kernel returned array of shape {K.shape}, "
            f"expected (..., {n_mean}, {n}, {n})"
        )
    if b.shape[-2:] != (n_mean, n):
        raise ValueError(
            f"rhs returned array of shape {b.shape}, expected (..., {n_mean}, {n})"
        )
    K[..., :, xp.arange(n), xp.arange(n)] = 0  # drop diagonal
    A = xp.eye(n, dtype=K.dtype, device=K.device) + K / n
    z_N_samples = (solve or xp.linalg.solve)(A, b[..., None])
    if z_N_samples.shape[-3:] != (n_mean, n, 1):
        raise ValueError(
            f"solve returned array of shape {z_N_samples.shape}, "
            f"expected (..., {n_mean}, {n}, 1)"
        )
    z_N_samples = z_N_samples[..., 0]  # (..., n_mean, n)

    def z_N(x: Array) -> Array:
        if x.shape[-1] != y.shape[-1]:
            raise ValueError(
                f"x has shape {x.shape}, expected (...(x), ..., {y.shape[-1]})"
            )
        K_x = kernel(x[..., None, None, :], y)  # (...(x), ..., n_mean, n)
        return (
            rhs(x) - xp.mean(xp.vecdot(xp.conj(K_x), z_N_samples, axis=-1), axis=-1) / n
        )

    return z_N
