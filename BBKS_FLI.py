# %% [markdown]
# ---
# title: "Field-level inference: Bayesion hierarchical model for joint field-parameter sampling"
# author: "Sacha Guerrini"
# date: today
# format: html
# jupyter: python3
# number-figures : true
# execute:
#   echo: false
# ---

# %%
#| echo: True
import numpy as np
import os

os.environ["JAX_PLATFORM_NAME"] = "gpu" #"cpu" if you don't have access to a GPU
import urllib.request
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as normal
import scipy.linalg as la
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

print(f"Device used by Jax: {jax.devices()[0]}")

np.random.seed(123456)

# %% [markdown]
"""
## Note

This document is based on a hands-on session given at [Les Houches summer school 2025 on the dark universe](https://indico.iap.fr/event/25/) on Field-Level Inference by Florent Leclercq.
The content of this document is based on the original correction notebook and aims to go further by applying Implicit Likelihood Inference to the model introduced here.
"""

# %% [markdown]
"""
## Note on the version

This document is generated using [`jax`](https://docs.jax.dev/en/latest/) and its sampling library [`blackjax`](https://blackjax-devs.github.io/blackjax/).
`jax` is a rapidly evolving framework. To reproduce this document, one should use the following versions of the software:

- `jax`: `0.5.0`
- `blackjax`: `1.2.5`
- `jaxili`: https://github.com/sachaguer/jaxili/tree/develop
"""


# %%
plt.rcParams.update({"lines.linewidth": 2})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update(
    {"text.latex.preamble": r"\usepackage{amsmath}\usepackage{upgreek}"}
)
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"font.size": 15})
dir = "./plots/BHM_field_parameter_sampling/"
os.makedirs(dir, exist_ok=True)

# %%
# Plotting utilities
colorsDict = {
    # Match pygtc up to v0.2.4
    "blues_old": ("#4c72b0", "#7fa5e3", "#b2d8ff"),
    "greens_old": ("#55a868", "#88db9b", "#bbffce"),
    "yellows_old": ("#f5964f", "#ffc982", "#fffcb5"),
    "reds_old": ("#c44e52", "#f78185", "#ffb4b8"),
    "purples_old": ("#8172b2", "#b4a5e5", "#37d8ff"),
    # New color scheme, dark colors match matplotlib v2
    "blues": ("#1f77b4", "#52aae7", "#85ddff"),
    "oranges": ("#ff7f0e", "#ffb241", "#ffe574"),
    "greens": ("#2ca02c", "#5fd35f", "#92ff92"),
    "reds": ("#d62728", "#ff5a5b", "#ff8d8e"),
    "purples": ("#9467bd", "#c79af0", "#facdff"),
    "browns": ("#8c564b", "#bf897e", "#f2bcb1"),
    "pinks": ("#e377c2", "#ffaaf5", "#ffddff"),
    "grays": ("#7f7f7f", "#b2b2b2", "#e5e5e5"),
    "yellows": ("#bcbd22", "#eff055", "#ffff88"),
    "cyans": ("#17becf", "#4af1ff", "#7dffff"),
}
defaultColorsOrder = [
    "blues",
    "oranges",
    "greens",
    "reds",
    "purples",
    "browns",
    "pinks",
    "grays",
    "yellows",
    "cyans",
]
colors = [colorsDict[cs] for cs in defaultColorsOrder]


def get_contours(Z, nBins=30, confLevels=(0.3173, 0.0455, 0.0027)):
    Z /= Z.sum()
    nContourLevels = len(confLevels)
    chainLevels = np.ones(nContourLevels + 1)
    histOrdered = np.sort(Z.flat)
    histCumulative = np.cumsum(histOrdered)
    nBinsFlat = np.linspace(0.0, nBins**2, nBins**2)

    for l in range(nContourLevels):
        # Find location of contour level in 1d histCumulative
        temp = np.interp(confLevels[l], histCumulative, nBinsFlat)
        # Find "height" of contour level
        chainLevels[nContourLevels - 1 - l] = np.interp(temp, nBinsFlat, histOrdered)

    return chainLevels


def get_contours_from_samples(
    samples_x,
    samples_y,
    weights=None,
    nBins=30,
    confLevels=(0.3173, 0.0455, 0.0027),
    smoothingKernel=1,
):
    from scipy.ndimage import gaussian_filter

    nContourLevels = len(confLevels)
    chainLevels = np.ones(nContourLevels + 1)
    extents = np.empty(4)

    # These are needed to compute the contour levels
    nBinsFlat = np.linspace(0.0, nBins**2, nBins**2)

    # Create 2d histogram
    if weights is None:
        weights = np.ones_like(samples_x)
    hist2d, xedges, yedges = np.histogram2d(
        samples_x, samples_y, weights=weights, bins=nBins
    )
    # image extent, needed below for contour lines
    extents = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # Normalize
    hist2d = hist2d / np.sum(hist2d)
    # Cumulative 1d distribution
    histOrdered = np.sort(hist2d.flat)
    histCumulative = np.cumsum(histOrdered)

    # Compute contour levels (from low to high for technical reasons)
    for l in range(nContourLevels):
        # Find location of contour level in 1d histCumulative
        temp = np.interp(confLevels[l], histCumulative, nBinsFlat)
        # Find "height" of contour level
        chainLevels[nContourLevels - 1 - l] = np.interp(temp, nBinsFlat, histOrdered)

    # Apply Gaussian smoothing
    contours = gaussian_filter(hist2d.T, sigma=smoothingKernel)

    xbins = (xedges[1:] + xedges[:-1]) / 2
    ybins = (yedges[1:] + yedges[:-1]) / 2

    return xbins, ybins, contours, chainLevels


# %%
# Download and define the Planck color map
from matplotlib.colors import ListedColormap

os.makedirs("data", exist_ok=True)

colormap_file = "data/Planck_Parchment_RGB.txt"
if not os.path.isfile("data/Planck_Parchment_RGB.txt"):
    url = "https://raw.githubusercontent.com/zonca/paperplots/master/data/Planck_Parchment_RGB.txt"
    urllib.request.urlretrieve(url, colormap_file)
planck = ListedColormap(np.loadtxt("data/Planck_Parchment_RGB.txt") / 255.0)
planck.set_bad("C7")  # color of missing pixels


# %% [markdown]

"""
## Context

In this exercise, we will illustrate field-level inference using the BBKS cosmological model.
We will focus on three parameters:

- $A_s$: the amplitude of the primordial power spectrum
- $n_s$: the spectral index of the primordial power spectrum
- $f_\mathrm{NL}$: a scalar parameter characterising the primordial non-Gaussianities

The primordial gravitational potential $\Phi_\mathrm{L}$ is a Gaussian random field with zero mean and power spectrum:

$$
P(k) = A_s k^{n_s - 1}.
$$

This primordial potential is mapped to a 'non-linear' potential field:

$$
\Phi_\mathrm{NL} = \Phi_\mathrm{L} + f_\mathrm{NL} \Phi_\mathrm{L}^2.
$$

The non-linear potential and the density contrast field are linked by a transfer function in Fourier space,

$$
\Phi_{\mathrm{NL}}(k) = \mathcal{F}[\Phi_{\mathrm{NL}}(x)], \quad \delta(k) = D_1 \sqrt{k} T(k) \phi_{\mathrm{NL}}(k),
$$

where $\mathcal{F}$ is the Fast Fourier Transform (FFT) operator, $T(k)$ is the transfer function, and $D_1$ is the linear growth factor (in arbitrary units).

The transfer function is modeled using:

$$
T(k) = \frac{\log(1 + \alpha q)}{\alpha q} \left(1 + \beta q + (\gamma q)^2 + (\delta q)^3 + (\epsilon q)^4\right)^{-1/4}, \quad q \equiv \frac{k}{\Gamma},
$$

with shape parameter $\Gamma = \Omega_m h \exp(-\Omega_b - \sqrt{2h}\Omega_b/\Omega_m)$. Other cosmological parameters are fixed to the following values:
"""

# %% [markdown]

# ### Problem parameters

# %%
# | echo: True
N = 32
L = 1.0  # box size
# The following parameters will be our fiducial values
A_s = 6e-9  # power spectrum normalisation, arbitrary units
n_s = 0.96  # spectral index
f_NL = 2000.0  # non-linear coupling parameter
D1 = 1.732e7  # growth factor of fluctuations at z=0, arbitrary units

# %% [markdown]

# ### Cosmological parameters

# %%
# | echo: True
Omega_b = 0.049
Omega_m = 0.315
h = 0.674
fb = Omega_b / Omega_m

# %% [markdown]

# ### BBKS parameters

# %%
# | echo: True

shape = Omega_m * h * np.exp(-Omega_b - np.sqrt(2.0 * h) * fb)
alpha = 2.34
beta = 3.89
gamma = 16.1
delta = 5.46
epsilon = 6.71

# %% [markdown]

"""
When $f_{\mathrm{NL}}=0$, the model is linear; therefore, $\delta(x)$ is a Gaussian random field with the BBKS power spectrum:

$$
P_\delta(k) = D_1^2 k T^2(k) P(k) = A_\mathrm{s} D_1^2 k^{n_\mathrm{s}}T^2(k).
$$

We model the data $d$ as a noisy observation of the $\delta$ field. We assume that the noise is zero-mean, additive and Gaussian, i.e.

$$
d(x) = \delta(x) + n(x), \quad n(x) \sim \mathcal{N}(0, N),
$$

where $N$ is the noise covariance matrix.

The figures below illustrate the model and the data generation process. It has been generated for fiducial parameters and will be used as observed data in what follows.
"""


# %%
# ## Setup physical model

# Write something here to explain the physical model...


def build_power_spectrum(N, A_s, n_s):
    # real-space grid spacing
    dx = L / N

    # build k-space grid for rfft2
    kx = jnp.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = jnp.fft.rfftfreq(N, d=dx) * 2 * np.pi
    kx, ky = jnp.meshgrid(kx, ky, indexing="ij")
    k = jnp.sqrt(kx**2 + ky**2)
    k = jnp.where(k == 0, 1.0, k)  # avoid division by zero

    # Build Pk safely (zero at k=0)
    Pk = A_s * k ** (n_s - 1.0)
    Pk = jnp.where(k == 0, 0, Pk)
    return Pk


@jax.jit
def phiL_from_real_noise(white_noise, A_s, n_s):
    """
    Given real space white noise, construct a Gaussian random field with power spectrum
    P(k) = A_s * k**(n_s - 1)
    Parameters
    ----------
        white_noise : real array of shape [N, N] (from jax.random.normal(key, (N, N)))
        A_s, n_s : parameters of the power spectrum

    Returns
    -------
        field: real array [N, N]
    """
    N = white_noise.shape[0]

    # Build the power spectrum
    Pkgrid = build_power_spectrum(N, A_s=A_s, n_s=n_s)

    # FFT the white_noise to k-space
    noise_k = jnp.fft.rfftn(white_noise)
    # Multiply by the square root of the power spectrum
    filtered_k = noise_k * jnp.sqrt(Pkgrid)
    # Inverse FFT back to real space
    field = jnp.fft.irfft2(filtered_k).real
    return field


def build_transfer_function(N):
    """
    Transfer function from primordial potential to density contrast.
    """
    # real-space grid spacing
    dx = L / N

    # build k-space grid for rfft2
    kx = jnp.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = jnp.fft.rfftfreq(N, d=dx) * 2 * np.pi
    kx, ky = jnp.meshgrid(kx, ky, indexing="ij")
    k = jnp.sqrt(kx**2 + ky**2)
    k = jnp.where(k == 0, 1.0, k)  # avoid division by zero

    # Build Tk safely (zero at k=0)
    T = np.zeros_like(k)

    # power spectrum, BBKS style
    q = k / shape
    aux = 1.0 + beta * q + (gamma * q) ** 2 + (delta * q) ** 3 + (epsilon * q) ** 4
    T = jnp.log(1.0 + alpha * q) / (alpha * q) * aux**-0.25
    T *= q ** (1.0 / 2.0)
    T = jnp.where(k == 0, 0.0, T)  # Set mean mode to zero transfer function

    return T


Tgrid = build_transfer_function(N)


@jax.jit
def delta_from_phi(phi, f_NL, D1=D1):
    r"""
    Given real-space $\Phi_\mathrm{L}$, compute $\Phi_\mathrm{NL}$ and $\delta$ such that:
    $$
    \delta(k) = D_1 T(k)^2 \Phi_\mathrm{NL}(k)
    $$
    and inverse FFT to real-space delta.

    Parameters
    ----------
        phi : real array of shape [N, N], the field Phi in real space
        f_NL : scalar, non-linear coupling parameter
        D1 : growth factor of fluctuations at z=0, arbitrary units
    Returns
    -------
        delta_real : real array of shape [N, N], the density contrast in real space
    """
    N = phi.shape[0]
    # 1. Construct Phi_NL
    Phi_NL = phi + f_NL * phi**2

    # 2. FFT to Fourier space
    Phi_NL_k = jnp.fft.rfft2(Phi_NL)

    # 3. Multiply by transfer function T(k)
    delta_k = D1 * Tgrid * Phi_NL_k
    delta_k = delta_k.at[0, 0].set(0.0)  # Explicitly set the mean mode to zero

    # 4. Inverse FFT to real-space
    delta_real = jnp.fft.irfft2(delta_k).real

    return delta_real


@jax.jit
def data_model(white_noise, A_s, n_s, f_NL, D1=D1):
    """
    Given real-space white noise, construct a Gaussian random field wit power spectrum
    P(k) = A_s * k**(n_s - 1) and compute the corresponding density contrast.

    Parameters
    ----------
        white_noise : real array of shape [N, N] (from jax.random.normal(key, (N, N)))
        A_s, n_s : parameters of the power spectrum
        f_NL : scalar, non-linear coupling parameter
        D1 : growth factor of fluctuations at z=0, arbitrary units

    Returns
    -------
        delta_real : real array of shape [N, N], the density contrast in real space
    """
    phi = phiL_from_real_noise(white_noise, A_s, n_s)
    delta = delta_from_phi(phi, f_NL, D1=D1)
    return delta


# %%
key = jax.random.PRNGKey(12)
white_noise = jax.random.normal(key, (N, N))
phi = phiL_from_real_noise(white_noise, A_s, n_s)
delta = delta_from_phi(phi, f_NL)


# %%
# |label: fig-primordial-potential
# |fig-cap: "Visualisation of the primordial potential and the density contrast field"
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)

phiNL = phi + f_NL * phi**2
# visualize the phi field
vmin = -max(-phi.min(), phi.max(), phiNL.min(), phiNL.max())
vmax = max(-phi.min(), phi.max(), phiNL.min(), phiNL.max())
im0 = ax0.imshow(phi, vmin=vmin, vmax=vmax, origin="lower", cmap=planck)
ax0.set_title("$\\Phi_\mathrm{L}$")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)

# visualise the phiNL field
im1 = ax1.imshow(phiNL, vmin=vmin, vmax=vmax, origin="lower", cmap=planck)
ax1.set_title("$\\Phi_\mathrm{NL}=\\Phi_\mathrm{L}+f_\mathrm{NL}\\Phi_\mathrm{L}^2$")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)

# visualise the density contrast field
im2 = ax2.imshow(
    delta,
    vmin=-max(-delta.min(), delta.max()),
    vmax=max(-delta.min(), delta.max()),
    origin="lower",
    cmap=planck,
)
ax2.set_title("Density contrast $\\delta$")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(im2, cax=cax2)

plt.show()


# %%
# Setup mask and noise covariance


def make_noise_field():
    """
    Build a 32x32 array "field" of noise variances:

     - Border of width 3 pixels everywhere: value = 1e0
     - Central 26x26 block: two regions of different noise variances
       - A low-noise region (1e-5)
       - A medium-noise region (5e-5) in the lower left corner
     - A 2x2 high-noise patch (1e0) centered in the array

    Returns
    -------
    field : ndarray, shape (32, 32)
        The resulting noise-variance field
    mask : ndarray, shape (32, 32)
        The resulting binary mask
    """
    field = np.zeros((N, N), dtype=float)
    mask = np.zeros((N, N), dtype=float)

    # 1) high-noise border
    bw = 3
    high = 1
    field[:bw, :] = high
    field[-bw:, :] = high
    field[:, :bw] = high
    field[:, -bw:] = high
    mask[:bw, :] = True
    mask[-bw:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True

    # 2) Central block (size 26x26)
    low = 1e-5
    med = 5e-5
    for i in range(bw, N - bw):
        for j in range(bw, N - bw):
            if i - j < -3:
                field[i, j] = med
            else:
                field[i, j] = low

    # 3) Sprinkle a 2x2 high-noise patch
    ci = 24
    cj = 24
    field[ci - 1 : ci + 1, cj - 1 : cj + 1] = high
    mask[ci - 1 : ci + 1, cj - 1 : cj + 1] = True

    return field, mask


# %%
noise_variance_field, mask = make_noise_field()
invN = np.diag(1.0 / noise_variance_field.flatten())
invN[np.where(invN <= 1)] = 0.0


# %%

# ### Generate mock data

# Data model: $d = \delta(s) + n

noise = np.random.normal(size=(N, N)) * np.sqrt(noise_variance_field)
noise_v = np.ma.masked_where(mask, noise)


# %%
data = delta + noise
data_v = np.ma.masked_where(mask, data)


# %%
# |label: fig-observation
# |fig-cap: "Visualisation of the observation. *Left*: the groundtruth density constrast, *Middle*: the mask and noise matrix, *Right*: the observed data."

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

# visualize the delta field
im0 = ax0.imshow(
    delta,
    vmin=-max(-delta.min(), delta.max()),
    vmax=max(-delta.min(), delta.max()),
    origin="lower",
    cmap=planck,
)
ax0.set_title("Groundtruth $\\delta$")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)

# visualize the noise field
vmin, vmax = np.min(noise_v), np.max(noise_v)
linthresh = 1e-5
linscale = 1.0
norm = SymLogNorm(
    linthresh=linthresh,
    linscale=linscale,
    vmin=-max(-vmin, vmax),
    vmax=max(-vmin, vmax),
    base=10,
)
cmap = plt.get_cmap("PiYG")
cmap.set_bad("C7")
im1 = ax1.imshow(noise_v, cmap=cmap, norm=norm, origin="lower")
ax1.set_title("Noise")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)

# visualize the data field
im2 = ax2.imshow(
    data_v,
    vmin=-max(-data_v.min(), data_v.max()),
    vmax=max(-data_v.min(), data_v.max()),
    origin="lower",
    cmap=planck,
)
ax2.set_title("Data")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(im2, cax=cax2)

plt.show()

# %% [markdown]
"""
## Inferring the parameters using the field

Using Bayesian inference, we want to estimate the parameters $\boldsymbol{\theta} = (A_\mathrm{s}, n_\mathrm{s}, f_\mathrm{NL})$ given the data $d$ and the noise variance field $N$.

To do so, we use Bayes' theorem:

$$
p(\boldsymbol{\theta} | d) \propto p(d | \boldsymbol{\theta}) p(\boldsymbol{\theta}).
$$

Here our data $d$ is the observed field, hence the name field-level inference. In the previous section, we have build a Bayesian Herarchical Model (BHM) for the data, which specifies the likelihood.

However, the likelihood is defined at the pixel level. We have access to $p(\boldsymbol{\theta}, z | d)$ where $z$ are the primordial pixel values. In practice, to get constraints on the cosmological parameters only, we must marginalise on pixel values $z$.

$$
p(\boldsymbol{\theta} | d) = \int p(\boldsymbol{\theta}, z | d) \, dz.
$$

This integral is intractable in practice. Sampling cosmological parameters and initial conditions is a challenging task as the problem is very high dimensional.
It requires sampling techniques that are not sensitive to the curse of dimensionality such as Hamiltonian Monte Carlo.

A first task is to write the log-prior, log-likelihood and log-posterior functions for the BHM.

We can summarise the model using the following diagram:

:::{.figure #fig-bhm}
```{mermaid}
flowchart TD
  A(("$$A_\mathrm{s} \sim \mathcal{N}(\mu_{A_\mathrm{s}}, \sigma_{A_\mathrm{s}})$$")) --> C["$$P(k)$$"]
  B(("$$n_\mathrm{s} \sim \mathcal{N}(\mu_{n_\mathrm{s}}, \sigma_{n_\mathrm{s}})$$")) --> C
  C --> D(("$$\Phi_\mathrm{L} \sim \mathcal{N}(0, P(k))$$"))
  D --> F["$$\Phi_\mathrm{NL}$$"]
  E(("$$f_\mathrm{NL} \sim \mathcal{N}(\mu_{f_\mathrm{NL}}, \sigma_{f_\mathrm{NL}})$$")) --> F
  F --> G["$$\delta$$"]
  G --> I["$$d$$"]
  H(("$$n \sim \mathcal{N}(0, N)$$")) --> I
  I <--> J[observation]
```
Bayesian hierarchical model for field-level inference. Round boxes are sampled from and rectangular boxes are derived.
:::

In what follows, we will use the following values for the prior of the cosmological parameters.
The likelihood is explicitely defined via the choices of distribution from which the parameters (prior) and latent variables are sampled from.
Probabilistic programming languages such as `pyro` or `numpyro` implement BHMs following the graph structure sketched in @fig-bhm.
"""


# %%
# | echo: True

# Define hyperparameters for the prior
A_s_scale = 6e-9
f_NL_scale = 2e3
mu_A_s = 0.9 * A_s_scale
sigma_A_s = 0.1 * A_s_scale
mu_n_s = 0.95
sigma_n_s = 0.01
mu_f_NL = 0.75 * f_NL_scale
sigma_f_NL = 0.25 * f_NL_scale

# %%


def A_s_from_theta(theta):
    """
    Extract A_s from the flat theta.
    """
    return theta[0] * A_s_scale


def n_s_from_theta(theta):
    """
    Extract n_s from the flat theta.
    """
    return theta[1]


def f_NL_from_theta(theta):
    """
    Extract f_NL from the flat theta.
    """
    return theta[2] * f_NL_scale


def field_from_theta(theta):
    """
    Extract the field from the flat theta.
    """
    return theta[3:].reshape((N, N))


def unpack_theta(theta, N):
    """
    Helper for unpacking the flat theta into params and field for debugging etc.
    """
    A_s = A_s_from_theta(theta)
    n_s = n_s_from_theta(theta)
    f_NL = f_NL_from_theta(theta)
    field = field_from_theta(theta)
    return A_s, n_s, f_NL, field


def pack_theta(A_s, n_s, f_NL, field):
    """
    Helper for packing the params and field into a flat theta.
    """
    theta = jnp.concatenate(
        (jnp.array([A_s / A_s_scale, n_s, f_NL / f_NL_scale]), field.flatten())
    )
    return theta


# %%


def log_prior(
    theta,
    mu_A_s=mu_A_s,
    sigma_A_s=sigma_A_s,
    mu_n_s=mu_n_s,
    sigma_n_s=sigma_n_s,
    mu_f_NL=mu_f_NL,
    sigma_f_NL=sigma_f_NL,
):
    """
    Separable Gaussian prior for (A_s, n_s, f_NL) parameters and white noise.

    Parameters
    ----------
    theta : [A_s, n_s, f_NL, signal...] array
        Signal is the white noise field.
    mu_A_s, sigma_A_s: mean and standard deviation for A_s
    mu_n_s, sigma_n_s: mean and standard deviation for n_s
    mu_f_NL, sigma_f_NL: mean and standard deviation for f_NL

    Returns
    -------
    log_prior: scalar, the log-prior value
    """
    A_s, n_s, f_NL, signal = unpack_theta(theta, N)

    # Compute the prior value for A_s, n_s, f_NL
    logp_A_s = normal.logpdf(A_s, loc=mu_A_s, scale=sigma_A_s)
    logp_n_s = normal.logpdf(n_s, loc=mu_n_s, scale=sigma_n_s)
    logp_f_NL = normal.logpdf(f_NL, loc=mu_f_NL, scale=sigma_f_NL)

    # Assuming a Gaussian prior with zero mean and unit variance for the signal
    logp_signal = -0.5 * jnp.sum(signal**2)

    return logp_signal + logp_A_s + logp_n_s + logp_f_NL


def log_likelihood(theta, data, noise_variance_field):
    """
    Compute the (unnormalised) log-likelihood of the data given the signal and noise variance field.

    Parameters
    ----------
    theta : [A_s, n_s, f_NL, signal...] array
        Signal is the white noise field.
    data : [N, N] array
        The observed data.
    noise_variance_field : [N, N] array
        The noise variance field

    Returns
    -------
        log_likelihood: scalar, the log-likelihood value
    """
    A_s, n_s, f_NL, signal = unpack_theta(theta, N)

    delta = data_model(signal, A_s=A_s, n_s=n_s, f_NL=f_NL)
    residual = data - delta
    # Gaussian likelihood at the pixel level
    log_likelihood = -0.5 * jnp.sum(residual**2 / noise_variance_field)
    return log_likelihood


def log_posterior(theta, data, noise_variance_field):
    """
    Compute the (unnormalised) log-posterior of the data given the signal and noise variance field.

    Parameters
    ----------
    theta : [A_s, n_s, f_NL, signal...] array
        Signal is the white noise field.
    data : [N, N] array
        The observed data.
    noise_variance_field : [N, N] array
        The noise variance field

    Returns
    -------
        log_posterior: scalar, the log-posterior value
    """
    log_likelihood_value = log_likelihood(theta, data, noise_variance_field)
    log_prior_value = log_prior(theta)
    return log_likelihood_value + log_prior_value


# %% [markdown]

# The output of the log-prior, log-likelihood and log-posterior functions is a scalar value and can be found for various examples below.
# %%
# | echo: True

# Test the functions with the ground truth white noise field
theta = pack_theta(A_s=A_s, n_s=n_s, f_NL=f_NL, field=white_noise)
log_prior(theta), log_likelihood(theta, data, noise_variance_field), log_posterior(
    theta, data, noise_variance_field
)


# %%
# | echo: True

# Generate a new white noise field and compute the log-likelihood
white_noise_2 = jax.random.normal(jax.random.PRNGKey(2), (N, N))
theta_2 = pack_theta(A_s=A_s, n_s=n_s, f_NL=f_NL, field=white_noise_2)
log_likelihood(theta_2, data, noise_variance_field), log_prior(theta_2), log_posterior(
    theta_2, data, noise_variance_field
)

# %%
# | echo: True

# Change the cosmological parameters and compute the log-likelihood
A_s_new = 7e-9
n_s_new = 0.98
f_NL_new = 2500.0
theta_3 = pack_theta(A_s=A_s_new, n_s=n_s_new, f_NL=f_NL_new, field=white_noise_2)
log_likelihood(theta_3, data, noise_variance_field), log_prior(theta_3), log_posterior(
    theta_3, data, noise_variance_field
)


# %%
# Compute the gradients of log_prior, log_likelihood, and log_posterior w.r.t. white noise using JAX autodiff
def d_log_prior_d_params(theta):
    """
    Compute the gradients of `log_prior` w.r.t all parameters.

    Parameters
    ----------
    theta : [A_s, n_s, f_NL, signal...] array
        Signal is the white noise field.

    Returns
    -------
     grad: [N*N+3] array, the gradient of log_prior w.r.t. all parameters
    """
    A_s, n_s, f_NL, white_noise = unpack_theta(theta, N)
    grad_A_s = (A_s - mu_A_s) / sigma_A_s**2
    grad_n_s = (n_s - mu_n_s) / sigma_n_s**2
    grad_f_NL = (f_NL - mu_f_NL) / sigma_f_NL**2
    grad_field = -white_noise
    grad = pack_theta(grad_A_s, grad_n_s, grad_f_NL, grad_field)
    return grad


def d_log_likelihood_d_params(theta, data, noise_variance_field):
    """
    Compute the gradient of `log_likelihood` w.r.t. all parameter using JAX autodiff.
    This function uses JAX's automatic differentiation to compute the gradient efficiently.

    Parameters
    ----------
    theta : [A_s, n_s, f_NL, signal...] array
        Signal is the white noise field.
    data : [N, N] array
        The observed data.
    noise_variance_field : [N, N] array
        The noise variance field

    Returns
    -------
    grad: [N*N+3] array, the gradient of log_likelihood w.r.t. all parameters
    """
    grad = jax.grad(log_likelihood)(theta, data, noise_variance_field)
    return grad


def d_log_posterior_d_params(theta, data, noise_variance_field):
    """
    Compute the gradient of `log_posterior` w.r.t. all parameter using JAX autodiff.
    This function uses JAX's automatic differentiation to compute the gradient efficiently.

    Parameters
    ----------
    theta : [A_s, n_s, f_NL, signal...] array
        Signal is the white noise field.
    data : [N, N] array
        The observed data.
    noise_variance_field : [N, N] array
        The noise variance field

    Returns
    -------
    grad: [N*N+3] array, the gradient of log_posterior w.r.t. all parameters
    """
    grad = jax.grad(log_posterior)(theta, data, noise_variance_field)
    return grad


# %% [markdown]
"""
### Sample the posterior using field-level inference (explicit)

To efficiently sample parameters in a high-dimensional space, one must rely on gradient-based Markov Chain Monte Carlo (MCMC) techniques such as HMC.
`jax` is a powerful library that uses autodiffertiation to compute gradient of functions in such a way that, given the log-likelihood function is written in `jax`, we have access to it for free.

In what follows, we use the No-U-Turn Sampler (NUTS) from the `blackjax` library, which is a JAX implementation of the NUTS algorithm to sample the cosmological parameters and the initial conditions of the field at the same time.
`jax` can run automatically on GPU. In this document, we ran on CPU on a laptop and sampling five chains of $10,000$ samples each takes around two hours. I have not checked the speed up obtained by running on GPU.

It is possible to speed up the sampling by using Gibbs sampling to sample the field with HMC and the cosmological parameters with e.g. slice sampling. The implementation of it currently does not perform correctly. In what follows, diagnostics of the chains will be presented using the NUTS algorithm.

**Why is it more efficient that way?**
"""

# %%

import blackjax
from typing import Any, Tuple


# Define the HMC sampling function
def sample_theta_hmc(
    theta_init: jnp.ndarray,
    data: jnp.ndarray,
    noise_variance_field: jnp.ndarray,
    n_samples: int,
    n_adapt: int = 1000,  # Number of adaptation steps
    rng_key: jax.random.PRNGKey = None,
) -> Tuple[jnp.ndarray, Any]:
    """
    Sample the posterior on the parameters and the white noise using Hamiltonian Monte Carlo.

    Parameters
    ----------
    theta_init : [A_s, n_s, f_NL, signal...] array
        Initial parameter values.
    data : [N, N] array
        The observed data.
    noise_variance_field : [N, N] array
        The noise variance field.
    n_samples : int
        Number of samples to draw.
    n_adapt : int
        Number of adaptation steps for the HMC sampler.
    rng_key : jax.random.PRNGKey
        Random key for reproducibility.

    Returns
    -------
    samples : [n_samples, N*N+3] array
        Samples from the posterior.
    states : Any
        Final states of the HMC sampler.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    N = data.shape[0]

    logprob = lambda flat_theta: log_posterior(flat_theta, data, noise_variance_field)

    initial_position = theta_init

    # Adaptation
    adapt = blackjax.window_adaptation(
        blackjax.nuts, logprob, target_acceptance_rate=0.8
    )

    (state, tuned_params), adaptation_state = adapt.run(
        rng_key, initial_position, num_steps=n_adapt
    )

    # Build the kernel for sampling
    kernel = blackjax.nuts(logprob, **tuned_params).step

    # Sampling loop
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    # Run the sampling
    states = inference_loop(rng_key, kernel, state, n_samples)

    # Reshape the samples to the original field shape
    samples = states.position

    return samples, states


# Define functions for Gibbs + HMC sampling


# Helper for univariate slice updates
def univariate_slice_sampler(logprob, x_init, rng_key, w=1.0, m=10, max_steps=50):
    logprob0 = logprob(x_init)
    e_key, l_key, r_key, i_key, _ = jax.random.split(rng_key, 5)
    y = logprob0 - jnp.abs(jax.random.exponential(e_key))
    u = jax.random.uniform(l_key)
    left = x_init - u * w
    right = left + w

    def step_out_left_fn(val):
        _left, _i = val
        return (logprob(_left) > y) & (_i < m)

    def step_out_left_body(val):
        _left, _i = val
        return (_left - w, _i + 1)

    left, _ = jax.lax.while_loop(step_out_left_fn, step_out_left_body, (left, 0))

    def step_out_right_fn(val):
        _right, _i = val
        return (logprob(_right) > y) & (_i < m)

    def step_out_right_body(val):
        _right, _i = val
        return (_right + w, _i + 1)

    right, _ = jax.lax.while_loop(step_out_right_fn, step_out_right_body, (right, 0))

    def body_fn(val):
        left, right, x, i, accepted = val
        new_key = jax.random.fold_in(i_key, i)
        x_new = jax.random.uniform(new_key) * (right - left) + left
        lp = logprob(x_new)
        cond = lp >= y
        left = jnp.where((~cond) & (x_new < x_init), x_new, left)
        right = jnp.where((~cond) & (x_new >= x_init), x_new, right)
        x = jnp.where(cond, x_new, x)
        i += 1
        return (left, right, x, i, accepted | cond)

    def cond_fn(val):
        left, right, x, i, accepted = val
        return (~accepted) & (i < max_steps)

    left, right, x, _, accepted = jax.lax.while_loop(
        cond_fn, body_fn, (left, right, x_init, 0, False)
    )
    return jnp.where(accepted, x, x_init)


def hmc_white_noise(theta, data, noise_variance_field, n_adapt, rng_key):
    Npix = len(theta) - 3
    N = int(np.sqrt(Npix))

    # logprob for field (fixed parameters)
    def logprob(flat_field):
        thet = jnp.concatenate([theta[:3], flat_field])
        return log_posterior(thet, data, noise_variance_field)

    initial_position = theta[3:]
    adapt = blackjax.window_adaptation(
        blackjax.hmc, logprob, num_integration_steps=10, target_acceptance_rate=0.8
    )

    (state, tuned_params), adaptation_state = adapt.run(
        rng_key, initial_position, num_steps=n_adapt
    )

    kernel = blackjax.hmc(logprob, **tuned_params).step

    # 1 HMC draw, return flat field
    key, subkey = jax.random.split(rng_key)
    state, _ = kernel(subkey, state)
    return state.position


def gibbs_sampler(
    theta_init,
    data,
    noise_variance_field,
    n_samples,
    n_adapt=20,
    rng_key=jax.random.PRNGKey(0),
):
    """
    Sample the posterior on the parameters and the white noise.
    Gibbs sampling is used for the cosmological parameters (A_s, n_s, f_NL),
    and Hamiltonian Monte Carlo (HMC) is used for the white noise field.

    Parameters
    ----------
    theta_init : [A_s, n_s, f_NL, signal...] array
        Initial parameter values.
    data : [N, N] array
        The observed data.
    noise_variance_field : [N, N] array
        The noise variance field.
    n_samples : int
        Number of samples to draw.
    n_adapt : int
        Number of adaptation steps for the HMC sampler.
    rng_key : jax.random.PRNGKey
        Random key for reproducibility.

    Returns
    -------
    samples : [n_samples, N*N+3] array
        Samples from the posterior.
    """
    Npix = len(theta_init) - 3

    def logprob_white_noise(field_flat, cosmo_params):
        theta = jnp.concatenate([cosmo_params, field_flat])
        return log_posterior(theta, data, noise_variance_field)

    # Adapt HMC step-size once
    field_init = theta_init[3:]
    cosmo_init = theta_init[:3]
    logprob = lambda field_flat: logprob_white_noise(field_flat, cosmo_init)

    hmc_adapt = blackjax.window_adaptation(
        blackjax.hmc,
        logprob,
        num_integration_steps=10,
        target_acceptance_rate=0.8,
    )
    # Use the adaptation parameters to setup the kernel at every Gibbs step
    (state, tuned_params), adaptation_state = hmc_adapt.run(
        rng_key, field_init, num_steps=n_adapt
    )

    # Helper: update each cosmo params by slice sampling
    def slice_update_var(idx, theta, rng_key):
        def logprob1d(x):
            return log_posterior(theta.at[idx].set(x), data, noise_variance_field)

        x_new = univariate_slice_sampler(logprob1d, theta[idx], rng_key)
        return theta.at[idx].set(x_new)

    # One step of Gibbs (scan body function)
    def gibbs_step(theta, rng_key):
        keys = jax.random.split(rng_key, 5)
        # Slice for each cosmological parameter
        theta = slice_update_var(0, theta, keys[0])  # Update A_s
        theta = slice_update_var(1, theta, keys[1])  # Update n_s
        theta = slice_update_var(2, theta, keys[2])  # Update f_NL
        # HMC for the field (using fixed HMC kernel)
        field_flat = theta[3:]
        state = blackjax.hmc.init(
            field_flat, lambda x: logprob_white_noise(x, theta[:3])
        )
        kernel = blackjax.hmc(logprob, **tuned_params).step
        state, _ = kernel(keys[3], state)
        theta = theta.at[3:].set(state.position)
        return theta, theta

    # Run scan
    key_seq = jax.random.split(rng_key, n_samples + 1)
    init_theta = theta_init
    _, samples = jax.lax.scan(gibbs_step, init_theta, key_seq[:-1])
    return samples  # shape: [n_samples, num_params + N**2]


# %%
n_samples = 10_000  # Number of samples to draw
N_chains = 5  # Number of chains to run
initial_scaling = 0.001  # Initial scaling for the white noise field
sampling_method = "hmc"  # Choose between 'gibbs' and 'hmc'
assert sampling_method in [
    "gibbs",
    "hmc",
], f"Unknown sampling method: {sampling_method}"


def draw_cosmo_theta0_from_prior(rng_key):
    from scipy.stats import norm

    # Draw A_s, n_s, f_NL from corresponding normals
    key1, key2, key3 = jax.random.split(rng_key, 3)
    A_s = norm.rvs(loc=mu_A_s, scale=sigma_A_s, random_state=int(key1[0]))
    n_s = norm.rvs(loc=mu_n_s, scale=sigma_n_s, random_state=int(key2[0]))
    f_NL = norm.rvs(loc=mu_f_NL, scale=sigma_f_NL, random_state=int(key3[0]))
    return [A_s, n_s, f_NL]


try:
    if sampling_method == "hmc":
        samples_chain = np.load(
            "data/BHM_field_parameters_sampling/samples_chain_hmc.npy",
            allow_pickle=True,
        ).item()
    else:
        samples_chain = np.load(
            "data/BHM_field_parameters_sampling/samples_chain_gibbs.npy",
            allow_pickle=True,
        ).item()
except FileNotFoundError:
    samples_chain = {}
    for c in range(N_chains):
        rng_key = jax.random.PRNGKey(42 + c)
        subkey1, subkey2 = jax.random.split(rng_key)
        theta0_cosmo = draw_cosmo_theta0_from_prior(subkey1)
        field_init = jax.random.normal(subkey2, (N, N)) * initial_scaling
        theta_init = pack_theta(*theta0_cosmo, field_init)
        if sampling_method == "hmc":
            samples_chain[c], infos = sample_theta_hmc(
                theta_init,
                data,
                noise_variance_field,
                n_samples=n_samples,
                n_adapt=2000,
                rng_key=rng_key,
            )
        else:
            samples_chain[c] = gibbs_sampler(
                theta_init,
                data,
                noise_variance_field,
                n_samples=n_samples,
                n_adapt=2000,
                rng_key=rng_key,
            )
    os.makedirs("data/BHM_field_parameters_sampling", exist_ok=True)
    if sampling_method == "hmc":
        np.save(
            "data/BHM_field_parameters_sampling/samples_chain_hmc.npy", samples_chain
        )
    else:
        np.save(
            "data/BHM_field_parameters_sampling/samples_chain_gibbs.npy", samples_chain
        )


# %% [markdown]

# #### Diagnose the chains

# ##### Log-likelihood

# %%

log_likelihoods_chain = {}
for c in range(N_chains):
    log_likelihoods_chain[c] = jax.vmap(log_likelihood, in_axes=(0, None, None))(
        samples_chain[c], data, noise_variance_field
    )


# %%
Nburnin = 100
n_thin_plot = 50  # Plot every nth element for clarity


# %%
# | label: fig-log-likelihood-trace
# | fig-cap: "Log-likelihood trace plot for each chain. The dashed line indicates a burn-in period of a hundred samples."
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_xlim(0, n_samples)
ax.set_ylim(450, 600)
ax.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
for c in range(N_chains):
    ax.plot(
        np.arange(0, n_samples, n_thin_plot),
        -log_likelihoods_chain[c][::n_thin_plot],
        label=f"Chain {c+1}",
    )
ax.axvline(Nburnin, color="k", linestyle=":", label="Burnt-in")
ax.set_xlabel("Sample index")
ax.set_ylabel("$-\\log \mathcal{L}$")
ax.set_title("Log-likelihood vs sample index")
ax.grid()
ax.legend()

plt.show()
# %% [markdown]

# ##### Trace plots

# We can then check the trace plots for parameters such pixels and cosmological parameters.

# %%
# | label: fig-trace-plot-pixel
# | fig-cap: "Trace plots for the pixel values at (10, 20) and (25, 10) for each chain. The dashed line indicates the groundtruth value."
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax0.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax0.set_ylabel("Pixel (10, 20)")
ax0.set_title("Trace plots for different chains")
for c in range(N_chains):
    ax0.plot(
        np.arange(0, n_samples, n_thin_plot),
        jax.vmap(field_from_theta)(samples_chain[c])[::n_thin_plot, 10, 20],
        marker=".",
    )
ax0.axhline(white_noise[10, 20], color="black", linestyle="--", label="Groundtruth")
ax0.axvline(Nburnin, color="black", linestyle=":")
ax0.legend()

ax1.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax1.set_ylabel("Pixel (25, 10)")
for c in range(N_chains):
    ax1.plot(
        np.arange(0, n_samples, n_thin_plot),
        jax.vmap(field_from_theta)(samples_chain[c])[::n_thin_plot, 25, 10],
        marker=".",
    )
ax1.axhline(white_noise[25, 10], color="black", linestyle="--", label="Groundtruth")
ax1.axvline(Nburnin, color="black", linestyle=":")
ax1.legend()

plt.show()


# %%
# | label: fig-trace-plot-cosmo
# | fig-cap: "Trace plots for the cosmological parameters $A_\\mathrm{s}$, $n_\\mathrm{s}$, and $f_\\mathrm{NL}$ for each chain. The dashed line indicates the groundtruth value and the shaded area the $2\\sigma$ region of the Gaussian prior."
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

ax0.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax0.set_ylabel("$A_\\mathrm{s}$")
ax0.set_title("Trace plots for different chains")
for c in range(N_chains):
    ax0.plot(
        np.arange(0, n_samples, n_thin_plot),
        jax.vmap(A_s_from_theta)(samples_chain[c])[::n_thin_plot],
        marker=".",
    )
ax0.axhline(mu_A_s, color="black", linestyle=":")
ax0.fill_between(
    np.arange(0, n_samples),
    np.ones(n_samples) * (mu_A_s + 2 * sigma_A_s),
    np.ones(n_samples) * (mu_A_s - 2 * sigma_A_s),
    color="black",
    alpha=0.1,
    label="Prior $2\\sigma$",
)
ax0.axhline(A_s, color="black", linestyle="--", label="Groundtruth")
ax0.axvline(Nburnin, color="black", linestyle=":")
ax0.legend(loc="lower left")

ax1.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax1.set_ylabel("$n_\\mathrm{s}$")
for c in range(N_chains):
    ax1.plot(
        np.arange(0, n_samples, n_thin_plot),
        jax.vmap(n_s_from_theta)(samples_chain[c])[::n_thin_plot],
        marker=".",
    )
ax1.axhline(mu_n_s, color="black", linestyle=":")
ax1.fill_between(
    np.arange(0, n_samples),
    np.ones(n_samples) * (mu_n_s + 2 * sigma_n_s),
    np.ones(n_samples) * (mu_n_s - 2 * sigma_n_s),
    color="black",
    alpha=0.1,
    label="Prior $2\\sigma$",
)
ax1.axhline(n_s, color="black", linestyle="--", label="Groundtruth")
ax1.axvline(Nburnin, color="black", linestyle=":")
ax1.legend(loc="lower left")

ax2.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax2.set_ylabel("$f_\\mathrm{NL}$")
for c in range(N_chains):
    ax2.plot(
        np.arange(0, n_samples, n_thin_plot),
        jax.vmap(f_NL_from_theta)(samples_chain[c])[::n_thin_plot],
        marker=".",
    )
ax2.axhline(mu_f_NL, color="black", linestyle=":")
ax2.fill_between(
    np.arange(0, n_samples),
    np.ones(n_samples) * (mu_f_NL + 2 * sigma_f_NL),
    np.ones(n_samples) * (mu_f_NL - 2 * sigma_f_NL),
    color="black",
    alpha=0.1,
    label="Prior $2\\sigma$",
)
ax2.axhline(f_NL, color="black", linestyle="--", label="Groundtruth")
ax2.axvline(Nburnin, color="black", linestyle=":")
ax2.legend(loc="lower left")

plt.show()


# %% [markdown]

# ##### Sequential posterior power spectrum

# Given our samples, we can do posterior predictive checks to further ensure that the chain has converged to the correct posterior.
# To do so, one needs to run the data model using the samples as input. We perform posterior predictive cheks by looking at the power spectrum of the reconstructred signal and the ground truth.
# In practice, the groundtruth signal will not be available but we can compare against perturbation theory predictions.


# %%

# Define a function to compute the power spectrum of the sampled delta fields
nbins = 200


@jax.jit
def power_spectrum_2d_jitted(field, L=1.0):
    # real-space grid spacing
    N = field.shape[0]
    dx = L / N

    # build k-space grid for rfft2
    kx = jnp.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = jnp.fft.rfftfreq(N, d=dx) * 2 * np.pi
    kx, ky = jnp.meshgrid(kx, ky, indexing="ij")
    k = jnp.sqrt(kx**2 + ky**2)
    kmag = k.flatten()

    # Compute rfft2
    field_k = jnp.fft.rfft2(field)
    power2d = (jnp.abs(field_k) ** 2) * dx**2 / (N**2)
    power_flat = power2d.flatten()

    # Bin edges (exclude k=0 when setting min)
    k_nonzero = kmag[1:] if kmag.size > 1 else kmag
    kmin = k_nonzero.min()
    kmax = kmag.max()
    bins = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nbins + 1)
    bin_idx = jnp.digitize(kmag, bins) - 1  # Get bin indices

    # Bin means
    Pk_sum = jnp.bincount(bin_idx, weights=power_flat, length=nbins)
    counts = jnp.bincount(bin_idx, length=nbins)
    Pk = jnp.where(counts > 0, Pk_sum / counts, 0.0)  # Avoid division by zero
    k_sum = jnp.bincount(bin_idx, weights=kmag, length=nbins)
    k_bin = jnp.where(counts > 0, k_sum / counts, 0.0)  # Mean k in each bin

    mask = counts > 0

    return k_bin, Pk, mask


def power_spectrum_2d(field, nbins=200, L=1.0):
    """
    Compute the 2D power spectrum of a 2D field using FFT and binning.

    Parameters
    ----------
    field : [N, N] array, the 2D field
    nbins : int, number of bins for the power spectrum
    L : float, box size

    Returns
    -------
    k_bin : [nbins] array, the binned wavenumbers
    Pk : [nbins] array, the binned power spectrum values
    """
    nbins = nbins
    k_bin, Pk, mask = power_spectrum_2d_jitted(field, L=L)
    return k_bin[mask], Pk[mask]


# %%
# Compute the power spectrum of the signal (white noise)

k_vals, Pk_signal_gt = power_spectrum_2d(white_noise, nbins=200)
k_vals, Pk_delta_gt = power_spectrum_2d(delta, nbins=200)

Nburnin = 1001
n_thin_PS = 50
Pk = np.zeros((Nburnin, len(k_vals)))
samples_chain1 = jax.vmap(field_from_theta)(samples_chain[0])
for i in range(0, Nburnin, n_thin_PS):
    k_vals, Pk[i] = power_spectrum_2d(samples_chain1[i])


# %%
# | label: fig-power-spectrum-signal
# | fig-cap: "Sequential posterior power spectrum of the reconstructed signal. The dashed line indicates the groundtruth power spectrum."
fig, (ax0, ax1) = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
)

ax0.loglog(
    k_vals, Pk_signal_gt, label="Groundtruth $s$", color="black", ls="--", zorder=5
)
cmap = plt.colormaps.get_cmap("winter")
norm = plt.Normalize(0, Nburnin - 1)

for i in range(0, Nburnin, n_thin_PS):
    color = cmap(norm(i))
    ax0.loglog(k_vals, Pk[i], color=color, alpha=0.6)

# Colorbar inside the top panel
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for showing the colorbar
cax = ax0.inset_axes([0.05, 0.25, 0.4, 0.03])  # [x0, y0, width, height]
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Sample index", labelpad=6, loc="center")
cax.xaxis.set_ticks_position("bottom")  # Ticks at the bottom (default for horizontal)

ax0.legend(loc=[0.05, 0.40])
ax0.set_ylabel(r"$P(k)$ [arbitrary units]")
ax0.set_title("Sequential posterior power spectrum of reconstructed signal")
ax0.grid()

ax1.loglog(k_vals, np.ones_like(k_vals), color="black", ls="--", zorder=5)
for i in range(0, Nburnin, n_thin_PS):
    color = cmap(norm(i))
    ax1.loglog(k_vals, Pk[i] / Pk_signal_gt, color=color, alpha=0.6)

ax1.set_xlabel(r"$k$ [2$\pi$/L]")
ax1.set_ylabel("$P(k)/P_\\mathrm{truth}(k)$")
ax1.grid()

plt.show()


# %%
# Reconstruct the delta field from the sampled white noise fields
def data_model_from_theta(theta):
    A_s, n_s, f_NL, signal = unpack_theta(theta, N)
    return data_model(signal, A_s=A_s, n_s=n_s, f_NL=f_NL)


delta_samples_chain1 = jax.vmap(data_model_from_theta)(samples_chain[0])

# Compute the power spectrum of the delta field samples
Pk_delta = np.zeros((Nburnin, len(k_vals)))
for i in range(0, Nburnin, n_thin_PS):
    k_vals, Pk_delta[i] = power_spectrum_2d(delta_samples_chain1[i])


# %%
#| label: fig-power-spectrum-delta
#| fig-cap: "Sequential posterior power spectrum of the reconstructed $\\delta$ field. The dashed line indicates the groundtruth power spectrum of the $\\delta$ field."
fig, (ax0, ax1) = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
)

ax0.loglog(
    k_vals, Pk_delta_gt, label="Groundtruth $\\delta$", color="black", ls="--", zorder=5
)
cmap = plt.colormaps.get_cmap("winter")
norm = plt.Normalize(0, Nburnin - 1)
Pk = np.zeros((Nburnin, len(k_vals)))

for i in range(0, Nburnin, n_thin_PS):
    color = cmap(norm(i))
    ax0.loglog(k_vals, Pk_delta[i], color=color, alpha=0.6)

# Colorbar inside the top panel
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for showing the colorbar
cax = ax0.inset_axes([0.05, 0.25, 0.4, 0.03])  # [x0, y0, width, height]
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Sample index", labelpad=6, loc="center")
cax.xaxis.set_ticks_position("bottom")  # Ticks at the bottom (default for horizontal)

ax0.legend(loc=[0.05, 0.40])
ax0.set_ylabel("$P_\\delta(k)$ [arbitrary units]")
ax0.set_title("Sequential posterior power spectrum of $\\delta$ fields")
ax0.grid()

ax1.loglog(k_vals, np.ones_like(k_vals), color="black", ls="--", zorder=5)
for i in range(0, Nburnin, n_thin_PS):
    color = cmap(norm(i))
    ax1.loglog(k_vals, Pk_delta[i] / Pk_delta_gt, color=color, alpha=0.6)

ax1.set_xlabel(r"$k$ [2$\pi$/L]")
ax1.set_ylabel("$P_\\delta(k)/P_{\\delta,\\mathrm{truth}}(k)$")
ax1.grid()

plt.show()


# %% [markdown]

# ##### Effective sample size in the chains

# When running an MCMC, samples are not independent. The effective sample size (ESS) is a measure of how many independent samples we have in our MCMC chain. It can be computed using the integrated autocorrelation time, which quantifies the correlation between samples at different lags.
# We here compute the ESS using function available in the [`emcee` documentation](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/).
# As the error of an MCMC estimator reduces with the number of samples, it is important to have a sufficiently large ESS.

# %%
# Based on python code from the emcee tutorials, https://emcee.readthedocs.io/en/stable/tutorials/autocorr/


def next_power_of_2(n: int) -> int:
    """Smallest power of two ≥ n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


# 1D autocorrelation function
def autocorr_func_1d(x, norm=True):
    """
    Compute the 1D autocorrelation via FFT in O(N log N).
    If norm=True, normalize so acf[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    nfft = 2 * next_power_of_2(n)

    # real FFT
    f = np.fft.rfft(x - np.mean(x), n=nfft)
    ps = (f * f.conjugate()).real  # power spectrum
    acf = np.fft.irfft(ps, n=nfft)[:n]
    acf /= 2 * nfft

    # normalise
    if norm:
        if acf[0] <= 0:
            return 0  # or raise ValueError("Autocorrelation function is zero or negative at lag 0.")
        else:
            acf /= acf[0]
    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    """
    Return the first lag k for which k < c * tau_k fails.
    If none fail, return len(taus)-1.
    """
    k = np.arange(len(taus))
    mask = k < c * taus
    # find first index where mask is False
    idx = np.argmax(~mask)
    return idx if mask[idx] == False else len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(x, c: float = 5.0) -> float:
    """
    Estimate the integrated autocorrelation time following
    Goodman & Weare (2010), with window parameter c.
    """
    acf = autocorr_func_1d(x, norm=True)
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def N_eff(x) -> float:
    """
    Effective number of independent samples in x.
    Accepts input as either a list or a 1D numpy array.
    """
    x = np.asarray(x)
    tau = autocorr_gw2010(x)
    if tau <= 0:
        return 0  # or raise ValueError("Autocorrelation time is zero or negative.")
    return x.size / tau


# %%
n_thin_corr = 200
n_steps = ((n_samples - 1) // n_thin_corr) + 1

# Preallocate arrays for Neff results
N_eff_10_20 = np.zeros((N_chains, n_steps))
N_eff_25_10 = np.zeros((N_chains, n_steps))
N_eff_A_s = np.zeros((N_chains, n_steps))
N_eff_n_s = np.zeros((N_chains, n_steps))
N_eff_f_NL = np.zeros((N_chains, n_steps))

# Vectorized extractors
field_10_20_vmap = jax.vmap(lambda theta: field_from_theta(theta)[10, 20])
field_25_10_vmap = jax.vmap(lambda theta: field_from_theta(theta)[25, 10])
A_s_vmap = jax.vmap(A_s_from_theta)
n_s_vmap = jax.vmap(n_s_from_theta)
f_NL_vmap = jax.vmap(f_NL_from_theta)

# For each chain, process all thinning windows
for c in range(N_chains):
    cur_chain = samples_chain[c]
    for i, samp in enumerate(np.arange(n_thin_corr, n_samples + 1, n_thin_corr)):
        this_slice = cur_chain[:samp]
        N_eff_10_20[c, i] = N_eff(field_10_20_vmap(this_slice))
        N_eff_25_10[c, i] = N_eff(field_25_10_vmap(this_slice))
        N_eff_A_s[c, i] = N_eff(A_s_vmap(this_slice))
        N_eff_n_s[c, i] = N_eff(n_s_vmap(this_slice))
        N_eff_f_NL[c, i] = N_eff(f_NL_vmap(this_slice))


# %%
# | label: fig-ess-pixels
# | fig-cap: "Effective sample size (ESS) for the pixels (10, 20) and (25, 10) in the field for each chain. The dashed line indicates the burn-in period."
import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax0.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax0.set_ylabel("Pixel (10,20)")
ax0.set_title("ESS for different chains")
for c in range(N_chains):
    ax0.plot(
        np.arange(0, n_samples, n_thin_corr),
        N_eff_10_20[c],
        label=f"Chain {c+1}",
        marker=".",
    )
ax0.axvline(Nburnin, color="black", linestyle=":")

ax1.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax1.set_xlim(0, n_samples)
ax1.set_xlabel("Sample index")
ax1.set_ylabel("Pixel (25,10)")
for c in range(N_chains):
    ax1.plot(
        np.arange(0, n_samples, n_thin_corr),
        N_eff_25_10[c],
        label=f"Chain {c+1}",
        marker=".",
    )
ax1.axvline(Nburnin, color="black", linestyle=":")
ax1.legend(loc="best")

plt.show()


# %%
# | label: fig-ess-cosmo
# | fig-cap: "Effective sample size (ESS) for the cosmological parameters $A_\\mathrm{s}$, $n_\\mathrm{s}$, and $f_\\mathrm{NL}$ for each chain. The dashed line indicates the burn-in period."
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

ax0.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax0.set_ylabel("$A_\\mathrm{s}$")
ax0.set_title("ESS for different chains")
for c in range(N_chains):
    ax0.plot(
        np.arange(0, n_samples, n_thin_corr),
        N_eff_A_s[c],
        label=f"Chain {c+1}",
        marker=".",
    )
ax0.axvline(Nburnin, color="black", linestyle=":")

ax1.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax1.set_xlim(0, n_samples)
ax1.set_ylabel("$n_\\mathrm{s}$")
for c in range(N_chains):
    ax1.plot(
        np.arange(0, n_samples, n_thin_corr),
        N_eff_n_s[c],
        label=f"Chain {c+1}",
        marker=".",
    )
ax1.axvline(Nburnin, color="black", linestyle=":")

ax2.set_prop_cycle(cycler("color", [plt.cm.Set2(i) for i in np.linspace(0, 1, 8)]))
ax2.set_xlim(0, n_samples)
ax2.set_ylabel("$f_\\mathrm{NL}$")
for c in range(N_chains):
    ax2.plot(
        np.arange(0, n_samples, n_thin_corr),
        N_eff_f_NL[c],
        label=f"Chain {c+1}",
        marker=".",
    )
ax2.axvline(Nburnin, color="black", linestyle=":")
ax2.set_xlabel("Sample index")
ax2.legend(loc="best")

plt.show()


# %% [markdown]
"""
##### Gelman-Rubin test

Another useful test is the Gelman-Rubin (GR) test. It assess if the samples are uncorrelated by computing the ratio between the variance within the chains and between multiple independant chains.

**Parameters**

* $m$: number of chains
* $n$: number of samples per chain

**Definitions**

* "between" chains variance:
\begin{equation}
B \equiv \frac{n}{m-1} \sum_{j=1}^m \left( \bar{\psi}_{. j} - \bar{\psi}_{..} \right)^2 \quad \mathrm{where} \quad \bar{\psi}_{. j} = \frac{1}{n} \sum_{i=1}^n \psi_{ij} \quad \mathrm{and} \quad \bar{\psi}_{..} = \frac{1}{m} \sum_{j=1}^m \bar{\psi}_{.j}
\end{equation}
* "within" chains variance:
\begin{equation}
W \equiv \frac{1}{m} \sum_{j=1}^m s_j^2 \quad \mathrm{where} \quad s_j^2 = \frac{1}{n-1} \sum_{i=1}^n \left( \psi_{ij} - \bar{\psi}_{.j} \right)^2
\end{equation}

**Estimators**:

Estimators of the marginal posterior variance of the estimand:

* $\widehat{\mathrm{var}}^- \equiv W$: underestimates the variance
* $\widehat{\mathrm{var}}^+ \equiv \frac{n - 1}{n}W + \frac{1}{n} B$: overestimates the variance

**Test**:

* Potential scale reduction factor: $\widehat{R} \equiv \sqrt{\frac{\widehat{\mathrm{var}}^+}{\widehat{\mathrm{var}}^-}}$
* Test: $\widehat{R} \rightarrow 1$ as $n \rightarrow \infty$
"""

# %%
def gelman_rubin(chain):
    # between chains variance
    Psi_dotj = np.mean(chain, axis=1)
    Psi_dotdot = np.mean(Psi_dotj, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1.0) * np.sum((Psi_dotj - Psi_dotdot) ** 2, axis=0)

    # within chains variance
    sj2 = np.var(chain, axis=1, ddof=1)
    W = np.mean(sj2, axis=0)

    # estimators
    var_minus = W
    var_plus = (n - 1.0) / n * W + 1.0 / n * B
    R_hat = np.sqrt(var_plus / var_minus)
    return R_hat


# %%
# | echo: True
# The Gelman-Rubin function expects (n_chains, n_samples, n_variates)
Rhat = gelman_rubin(jnp.array([samples_chain[c] for c in range(N_chains)]))

# Run Gelman-Rubin
Rhat_A_s = Rhat[0]
Rhat_n_s = Rhat[1]
Rhat_fNL = Rhat[2]
Rhat_field_flat = Rhat[3:]  # shape: (N*N,)
Rhat_field = Rhat_field_flat.reshape((N, N))

print("Gelman-Rubin stats:")
print("A_s:", Rhat_A_s)
print("n_s:", Rhat_n_s)
print("fNL:", Rhat_fNL)


# %%
# | label: fig-gelman-rubin
# | fig-cap: "Gelman-Rubin statistic $\\hat{R}$ for the field."
# Visualize the Gelman-Rubin statistic
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(Rhat_field, vmin=1, vmax=Rhat_field.max(), origin="lower", cmap="bone_r")
ax.set_title("Gelman-Rubin statistic $\hat{R}$")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im, cax=cax)
plt.tight_layout()

plt.show()


# %% [markdown]
"""
### Visualise summaries of the chains

Now that we have carefully checked the convergence of the MCMC chains, we can visualise the results.

First, we can use the samples to reconstruct the $\delta$ field and check that it visually matches the truth.
"""



# %%
# Join the different chains into one array for the posterior samples
samples = jnp.concatenate([samples_chain[c] for c in range(N_chains)], axis=0)
white_noise_samples = jax.vmap(field_from_theta)(samples)
A_s_samples = jax.vmap(A_s_from_theta)(samples)
n_s_samples = jax.vmap(n_s_from_theta)(samples)
f_NL_samples = jax.vmap(f_NL_from_theta)(samples)
PhiL = phiL_from_real_noise(white_noise, A_s, n_s)
PhiL_samples = jax.vmap(phiL_from_real_noise)(
    white_noise_samples, A_s_samples, n_s_samples
)
PhiNL = PhiL + f_NL * PhiL**2


def phiNL_from_phiL(phiL, f_NL):
    return phiL + f_NL * phiL**2


PhiNL_samples = jax.vmap(phiNL_from_phiL)(PhiL_samples, f_NL_samples)
delta_samples = jax.vmap(data_model)(
    white_noise_samples, A_s_samples, n_s_samples, f_NL_samples
)


# %%
@jax.jit
def compute_summaries(samples):
    empirical_mean = jnp.mean(samples, axis=0)
    empirical_var = jnp.var(samples, axis=0)
    return empirical_mean, empirical_var


empirical_mean, empirical_var = compute_summaries(white_noise_samples)


# %%
PhiL_mean, PhiL_var = compute_summaries(PhiL_samples)
PhiNL_mean, PhiNL_var = compute_summaries(PhiNL_samples)
delta_mean, delta_var = compute_summaries(delta_samples)


# %%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.25)

# visualize the signal field
im0 = ax0.imshow(
    white_noise,
    vmin=-max(-white_noise.min(), white_noise.max()),
    vmax=max(-white_noise.min(), white_noise.max()),
    origin="lower",
    cmap=planck,
)
ax0.set_title("Initial conditions")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar0 = fig.colorbar(im0, cax=cax0)

# visualize the empirical mean of Wiener filter samples
im1 = ax1.imshow(
    empirical_mean,
    vmin=-max(-white_noise.min(), white_noise.max()),
    vmax=max(-white_noise.min(), white_noise.max()),
    origin="lower",
    cmap=planck,
)
ax1.set_title("Empirical mean of samples")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar1 = fig.colorbar(im1, cax=cax1)

# visualize the empirical variance of Wiener filter samples
im2 = ax2.imshow(
    empirical_var,
    norm=LogNorm(vmin=empirical_var.min(), vmax=empirical_var.max()),
    origin="lower",
    cmap="Greys",
)
ax2.set_title("Empirical variance of samples")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
cbar2 = fig.colorbar(im2, cax=cax2)

plt.show()


# %%
#| label: fig-reconstruction
#| fig-cap: "Reconstruction of the $\\Phi_\\mathrm{L}$, $\\Phi_\\mathrm{NL}$, and $\\delta$ fields from the initial conditions and cosmological parameters samples. The first row shows the initial conditions, the second row the groundtruth fields, the third row shows the empirical mean of the samples, and the fourth row shows the empirical variance of the samples."
fig, ((ax0a, ax1a, ax2a), (ax0b, ax1b, ax2b), (ax0c, ax1c, ax2c)) = plt.subplots(
    3, 3, figsize=(18, 16)
)
plt.subplots_adjust(wspace=0.25)

# visualize the groundtruth PhiL field
vmin = -max(-phi.min(), phi.max(), phiNL.min(), phiNL.max())
vmax = max(-phi.min(), phi.max(), phiNL.min(), phiNL.max())
im0a = ax0a.imshow(PhiL, vmin=vmin, vmax=vmax, origin="lower", cmap=planck)
ax0a.set_title("Groundtruth $\\Phi_\mathrm{L}$")
divider = make_axes_locatable(ax0a)
cax0a = divider.append_axes("right", size="5%", pad=0.1)
cbar0a = fig.colorbar(im0a, cax=cax0a)

# visualize the empirical mean of Wiener filter PhiL samples
im1a = ax1a.imshow(PhiL_mean, vmin=vmin, vmax=vmax, origin="lower", cmap=planck)
ax1a.set_title("Empirical mean of $\\Phi_\mathrm{L}$ samples")
divider = make_axes_locatable(ax1a)
cax1a = divider.append_axes("right", size="5%", pad=0.1)
cbar1a = fig.colorbar(im1a, cax=cax1a)

# visualize the empirical variance of Wiener filter PhiL samples
im2a = ax2a.imshow(
    PhiL_var,
    norm=LogNorm(vmin=PhiL_var.min(), vmax=PhiL_var.max()),
    origin="lower",
    cmap="Greys",
)
ax2a.set_title("Empirical variance of $\\Phi_\mathrm{L}$ samples")
divider = make_axes_locatable(ax2a)
cax2a = divider.append_axes("right", size="5%", pad=0.1)
cbar2a = fig.colorbar(im2a, cax=cax2a)

# visualize the groundtruth PhiNL field
im0b = ax0b.imshow(PhiNL, vmin=vmin, vmax=vmax, origin="lower", cmap=planck)
ax0b.set_title("Groundtruth $\\Phi_\\mathrm{NL}$")
divider = make_axes_locatable(ax0b)
cax0b = divider.append_axes("right", size="5%", pad=0.1)
cbar0b = fig.colorbar(im0b, cax=cax0b)

# visualize the empirical mean of Wiener filter PhiNL samples
im1b = ax1b.imshow(PhiNL_mean, vmin=vmin, vmax=vmax, origin="lower", cmap=planck)
ax1b.set_title("Empirical mean of $\\Phi_\\mathrm{NL}$ samples")
divider = make_axes_locatable(ax1b)
cax1b = divider.append_axes("right", size="5%", pad=0.1)
cbar1b = fig.colorbar(im1b, cax=cax1b)

# visualize the empirical variance of Wiener filter PhiNL samples
im2b = ax2b.imshow(
    PhiNL_var,
    norm=LogNorm(vmin=PhiNL_var.min(), vmax=PhiNL_var.max()),
    origin="lower",
    cmap="Greys",
)
ax2b.set_title("Empirical variance of $\\Phi_\\mathrm{NL}$ samples")
divider = make_axes_locatable(ax2b)
cax2b = divider.append_axes("right", size="5%", pad=0.1)
cbar2b = fig.colorbar(im2b, cax=cax2b)

# visualize the groundtruth delta field
im0c = ax0c.imshow(
    delta,
    vmin=-max(-delta.min(), delta.max()),
    vmax=max(-delta.min(), delta.max()),
    origin="lower",
    cmap=planck,
)
ax0c.set_title("Groundtruth $\\delta$")
divider = make_axes_locatable(ax0c)
cax0c = divider.append_axes("right", size="5%", pad=0.1)
cbar0c = fig.colorbar(im0c, cax=cax0c)

# visualize the empirical mean of Wiener filter delta samples
im1c = ax1c.imshow(
    delta_mean,
    vmin=-max(-delta.min(), delta.max()),
    vmax=max(-delta.min(), delta.max()),
    origin="lower",
    cmap=planck,
)
ax1c.set_title("Empirical mean of $\\delta$ samples")
divider = make_axes_locatable(ax1c)
cax1c = divider.append_axes("right", size="5%", pad=0.1)
cbar1c = fig.colorbar(im1c, cax=cax1c)

# visualize the empirical variance of Wiener filter delta samples
im2c = ax2c.imshow(
    delta_var,
    norm=LogNorm(vmin=delta_var.min(), vmax=delta_var.max()),
    origin="lower",
    cmap="Greys",
)
ax2c.set_title("Empirical variance of $\\delta$ samples")
divider = make_axes_locatable(ax2c)
cax2c = divider.append_axes("right", size="5%", pad=0.1)
cbar2c = fig.colorbar(im2c, cax=cax2c)

plt.show()


# %%
def gaussian2d_grid(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=0.0):
    """
    Returns a normalized Gaussian density grid evaluated on (x, y).
    x, y = meshgrid arrays.
    """
    X = x - mu_x
    Y = y - mu_y
    z = (
        X**2 / sigma_x**2 + Y**2 / sigma_y**2 - 2 * rho * X * Y / (sigma_x * sigma_y)
    ) / (2 * (1 - rho**2))
    norm = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    return norm * np.exp(-z)


nBins = 50

A_s_grid = np.linspace(mu_A_s - 4 * sigma_A_s, mu_A_s + 4 * sigma_A_s, nBins)
n_s_grid = np.linspace(mu_n_s - 4 * sigma_n_s, mu_n_s + 4 * sigma_n_s, nBins)
f_NL_grid = np.linspace(mu_f_NL - 4 * sigma_f_NL, mu_f_NL + 4 * sigma_f_NL, nBins)

A_s_mesh, n_s_mesh = np.meshgrid(A_s_grid, n_s_grid, indexing="ij")
prior_density_A_s_n_s = gaussian2d_grid(
    A_s_mesh, n_s_mesh, mu_A_s, mu_n_s, sigma_A_s, sigma_n_s
)
prior_chainLevels_A_s_n_s = get_contours(prior_density_A_s_n_s, nBins=nBins)

A_s_mesh, f_NL_mesh = np.meshgrid(A_s_grid, f_NL_grid, indexing="ij")
prior_density_A_s_f_NL = gaussian2d_grid(
    A_s_mesh, f_NL_grid, mu_A_s, mu_f_NL, sigma_A_s, sigma_f_NL
)
prior_chainLevels_A_s_f_NL = get_contours(prior_density_A_s_f_NL, nBins=nBins)

n_s_mesh, f_NL_mesh = np.meshgrid(n_s_grid, f_NL_grid, indexing="ij")
prior_density_n_s_f_NL = gaussian2d_grid(
    n_s_mesh, f_NL_grid, mu_n_s, mu_f_NL, sigma_n_s, sigma_f_NL
)
prior_chainLevels_n_s_f_NL = get_contours(prior_density_n_s_f_NL, nBins=nBins)

# %% [markdown]

# We can finally visualise the posterior samples in the cosmological parameters space.

# %%
#| label: fig-cosmo-params
#| fig-cap: "Posterior samples in the cosmological parameters space. The blue contours show the prior distribution, while the green filled contours show the posterior distribution. The lines indicate the one, two, and three $\\sigma$ confidence level. The dashed lines indicate the groundtruth."
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

n_thin_scatter = 5

ax0 = fig.add_subplot(2, 2, 1)
xbins, ybins, contours, chainLevels = get_contours_from_samples(
    A_s_samples, n_s_samples
)
nContourLevels = len(chainLevels)
ax0.contourf(
    A_s_grid,
    n_s_grid,
    prior_density_A_s_n_s,
    levels=prior_chainLevels_A_s_n_s,
    colors=colors[0][:nContourLevels][::-1],
)
ax0.contour(
    A_s_grid,
    n_s_grid,
    prior_density_A_s_n_s,
    levels=prior_chainLevels_A_s_n_s,
    colors=colors[0][:nContourLevels][::-1],
)
ax0.scatter(
    A_s_samples[::n_thin_scatter], n_s_samples[::n_thin_scatter], color="black", s=2
)
ax0.contourf(
    xbins,
    ybins,
    contours,
    levels=chainLevels,
    colors=colors[2][:nContourLevels][::-1],
    alpha=0.5,
)
ax0.contour(
    xbins, ybins, contours, levels=chainLevels, colors=colors[2][:nContourLevels][::-1]
)
ax0.axhline(n_s, color="black", ls="--", lw=1)
ax0.axvline(A_s, color="black", ls="--", lw=1)
ax0.set_xlabel("$A_\\mathrm{s}$")
ax0.set_ylabel("$n_\\mathrm{s}$")

prior_color = colors[0][1]
post_color = colors[2][1]
legend_elements = [
    mlines.Line2D(
        [], [], color=prior_color, linestyle="solid", linewidth=2, label="Prior"
    ),
    mlines.Line2D(
        [], [], color=post_color, linestyle="solid", linewidth=2, label="Posterior"
    ),
    mlines.Line2D(
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=2,
        label="Samples",
    ),
    mlines.Line2D(
        [], [], color="black", linestyle="--", linewidth=1, label="Groundtruth"
    ),
]
ax0.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.12, 0.36),
    loc="upper left",
    borderaxespad=0,
)

ax1 = fig.add_subplot(2, 2, 3)
xbins, ybins, contours, chainLevels = get_contours_from_samples(
    A_s_samples, f_NL_samples
)
nContourLevels = len(chainLevels)
ax1.contourf(
    A_s_grid,
    f_NL_grid,
    prior_density_A_s_f_NL,
    levels=prior_chainLevels_A_s_f_NL,
    colors=colors[0][:nContourLevels][::-1],
)
ax1.contour(
    A_s_grid,
    f_NL_grid,
    prior_density_A_s_f_NL,
    levels=prior_chainLevels_A_s_f_NL,
    colors=colors[0][:nContourLevels][::-1],
)
ax1.scatter(A_s_samples, f_NL_samples, color="black", s=2)
ax1.contourf(
    xbins,
    ybins,
    contours,
    levels=chainLevels,
    colors=colors[2][:nContourLevels][::-1],
    alpha=0.5,
)
ax1.contour(
    xbins, ybins, contours, levels=chainLevels, colors=colors[2][:nContourLevels][::-1]
)
ax1.axhline(f_NL, color="black", ls="--", lw=1)
ax1.axvline(A_s, color="black", ls="--", lw=1)
ax1.set_xlabel("$A_\\mathrm{s}$")
ax1.set_ylabel("$f_\\mathrm{NL}$")

ax2 = fig.add_subplot(2, 2, 4)
xbins, ybins, contours, chainLevels = get_contours_from_samples(
    n_s_samples, f_NL_samples
)
nContourLevels = len(chainLevels)
ax2.contourf(
    n_s_grid,
    f_NL_grid,
    prior_density_n_s_f_NL,
    levels=prior_chainLevels_n_s_f_NL,
    colors=colors[0][:nContourLevels][::-1],
)
ax2.contour(
    n_s_grid,
    f_NL_grid,
    prior_density_n_s_f_NL,
    levels=prior_chainLevels_n_s_f_NL,
    colors=colors[0][:nContourLevels][::-1],
)
ax2.scatter(n_s_samples, f_NL_samples, color="black", s=2)
ax2.contourf(
    xbins,
    ybins,
    contours,
    levels=chainLevels,
    colors=colors[2][:nContourLevels][::-1],
    alpha=0.5,
)
ax2.contour(
    xbins, ybins, contours, levels=chainLevels, colors=colors[2][:nContourLevels][::-1]
)
ax2.axhline(f_NL, color="black", ls="--", lw=1)
ax2.axvline(n_s, color="black", ls="--", lw=1)
ax2.set_xlabel("$n_\\mathrm{s}$")
ax2.set_ylabel("$f_\\mathrm{NL}$")


plt.show()


# %% [markdown]

# This result has been obtained sampling all latent variables. It is also possible to extract information at the field-level without relying on high-dimensional sampling of the initial conditions.
