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
) -> Callable[[Array], Array]:
    r"""
    Solve integral equations of the second kind of the following form.

    $\forall d \in \mathbb{N}.
    \forall \Omega \in \mathbb{R}^d [\Omega \text{is Lipschitz}].
    \forall p \in L^\infty(\Omega, \mathbb{R}_{\geq 0}) [\int_\Omega p(y) dy = 1].
    \forall f, z \in L^2(\Omega,\mathbb{C}).
    \forall k \in L^2(\Omega,L^2(\Omega,\mathbb{C})
    [z(y) + \int k(y, y') z(y') p(y') dy' = f(y)].
    $

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

    Returns
    -------
    Callable[[Array], Array]
        Approximate solution function z_N that takes an array of shape (...(x), ..., d)
        and returns an array of shape (...(x), ...).

    """
    y = random_samples(n * n_mean)
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
    A = xp.eye(n) + K / n
    z_N_samples = xp.linalg.solve(A, b[..., None])[..., 0]  # (..., n_mean, n)

    def z_N(x: Array) -> Array:
        K_x = kernel(x[..., None, None, :], y)  # (...(x), ..., n_mean, n)
        print(K_x.shape)
        return rhs(x) - xp.mean(xp.sum(K_x * z_N_samples, axis=-1), axis=-1) / n

    return z_N
