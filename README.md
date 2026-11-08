# Monte-Carlo Nystrom

<p align="center">
  <a href="https://github.com/ultrasphere-dev/montecarlo-nystrom/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/ultrasphere-dev/montecarlo-nystrom/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://montecarlo-nystrom.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/montecarlo-nystrom.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
</p>
<p align="center">
  <a href="https://codecov.io/gh/ultrasphere-dev/montecarlo-nystrom">
    <img src="https://img.shields.io/codecov/c/github/ultrasphere-dev/montecarlo-nystrom.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/j178/prek">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json" alt="prek">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/montecarlo-nystrom/">
    <img src="https://img.shields.io/pypi/v/montecarlo-nystrom.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/montecarlo-nystrom.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/montecarlo-nystrom.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://montecarlo-nystrom.readthedocs.io" target="_blank">https://montecarlo-nystrom.readthedocs.io </a>

**Source Code**: <a href="https://github.com/ultrasphere-dev/montecarlo-nystrom" target="_blank">https://github.com/ultrasphere-dev/montecarlo-nystrom </a>

---

Monte-Carlo Nystrom method in NumPy / PyTorch

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install montecarlo-nystrom
```

## Usage

Solve integral equations of the second kind of the following form.

$\forall d \in \mathbb{N}.$
$\forall \Omega \in \mathbb{R}^d [\Omega \text{ is bounded Lipschitz}].$
$`\forall p \in L^\infty(\Omega, {\mathbb{R}}_{\geq 0}) [\int_\Omega p(y) dy = 1].`$
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

The below example solves the case where $d = 1$, $\Omega = [0, 1]$, $p$ (`random_samples`) is the uniform distribution on $[0, 1]$, $k(x, y) = \|x - y\|^{-0.4}$ (`kernel`), and $f(x) = 1$ (`rhs`), and evaluates the solution at $x = (0.5,)$.

```python
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
```

## References

- Feppon, F., & Ammari, H. (2022). Analysis of a Monte-Carlo Nystrom Method. SIAM J. Numer. Anal. Retrieved from https://epubs.siam.org/doi/10.1137/21M1432338
