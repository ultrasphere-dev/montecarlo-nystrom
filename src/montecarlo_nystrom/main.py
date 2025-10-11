from collections.abc import Callable

from array_api._2024_12 import Array
from array_api_compat import array_namespace


def montecarlo_nystrom(
    *,
    random_samples: Callable[[int], Array],
    kernel: Callable[[Array, Array], Array],
    rhs: Callable[[Array], Array],
    n: int,
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
    rhs : Callable[[Array], Array]
        Right-hand side function f of (..., :, d) -> (..., :).
    n : int
        The number of random samples to draw from the distribution p.

    Returns
    -------
    Callable[[Array], Array]
        Approximate solution function z_N that takes an array of shape (...(x), ..., d)
        and returns an array of shape (...(x), ...).

    """
    y = random_samples(n)
    K = kernel(y[..., :, None, :], y[..., None, :, :])
    b = rhs(y)
    xp = array_namespace(K, b)
    A = xp.eye(n) + K / n
    z_N_samples = xp.linalg.solve(A, b)

    def z_N(x: Array) -> Array:
        K_x = kernel(x[..., None, :], y)
        return rhs(x) - K_x @ z_N_samples / n

    return z_N
