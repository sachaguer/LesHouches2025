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

os.environ["JAX_PLATFORM_NAME"] = "cpu" #"cpu" if you don't have access to a GPU
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

"""
## Implicit Likelihood Inference (ILI) - Simulation-Based Inference (SBI)

In the previous section, focusing on explicit likelihood inference, we built a model where the likelihood is tractable at the pixel level.
In return, we had to used advanced sampling techniques requiring to differentiate the likelihood. Library like `jax` make our lives easier with this kind of tasks but we do not necessarily need it to estimate the cosmological parameters.

Indeed, the model being fully specified, we can sample from it and use those samples to build the posterior distribution of the cosmological parameters. We call this approach 'implicit' likelihood inference because we do not need to explictly build the likelihood.
As we will see, we can solely rely on a blackbox that brings us from the cosmological parameters to some observation to make this technique work.
To do so, ILI relies on Deep Learning generative models to learn the posterior $p(\boldsymbol{\theta}|d)$ of interest.
However, there is no free lunch. The generated field is still high dimensional so we need to perform some compression of the field. A vanilla way of compressing the field is to compute the power spectrum, but there are
deep learning techniques that allow to compress the field in a low dimensional space while preserving the information.

In this section, we will apply ILI/SBI techniques to the toy model introduced in this document to show:

- how to use ILI/SBI to estimate the cosmological parameters.
- that ILI/SBI will obtain the same constraints than explicit inference as long as the model is the same.
- compare compression techniques and assess their optimality.

To perform this task, we will use the [`jaxili` library](https://jaxili.readthedocs.io/en/latest/manual/) to train the neural network and compressor.
"""

# %%
from jaxili.inference import NPE, NLE

# %% [markdown]

"""
![A summary of the SBI approach for cosmology. Credit: SimBIG Collaboration.](https://changhoonhahn.github.io/simbig/current/_images/graphics.jpg){#fig-sbi-in-a-nutshell}
"""

# %% [markdown]

"""
The approach for ILI is summarised in @fig-sbi-in-a-nutshell. We wrote functions to generate/simulate an observation (our $32 \times 32$ masked and noisy field) from a stochastic process depending on cosmological parameters.
We now want to estimate the cosmological parameters and their uncertainty so the idea is to find the cosmological parameters that produce observations that 'look like' our observation.
Contrary to explicit inference, we completely lose track of the initial conditions but their uncertainty is accounted for as the stochastic process used to generate the simulations accounts for it.

A clever way to find which cosmological parameters produce output that 'looks like' our observation is to use Deep Learning generative models. Those models are well suited to learn distributions which in our context can be associated to the posterior $p(\boldsymbol{\theta} | d)$ (Neural Posterior Estimation) or the likelihood $p(d | \boldsymbol{\theta})$ (Neural Likelihood Estimation).
We use neural networks called Normalizing Flows (NFs) to learn those distribution. NFs are very suitable as they build several transport layers from a simple distribution to our target distribution.
Albeit some design criteria, it makes the evaluation of the log-probability and sampling easy and fast to do.

We first need to define a function that takes cosmological parameters as input and produced observations. We will call this function our `simulator`.
"""



# %%
# | echo: true

def simulator(cosmo_params, rng_key):
    """
    Simulator function generating a field observable accounting for all sources of stochasticity.

    Parameters
    ----------
    cosmo_params : [A_s, n_s, f_NL] array
        The cosmological parameters
    rng_key : jax.random.PRNGKey
        The key for randomness
    """
    A_s, n_s, f_NL = cosmo_params

    #1. Generate some white noise
    subkey, key = jax.random.split(rng_key)
    white_noise = jax.random.normal(subkey, shape=(N, N))

    #2. Generate the delta field
    delta = data_model(white_noise, A_s=A_s, n_s=n_s, f_NL=f_NL)
    #Note: data_model is a jitted function so this step should be quite fast

    #3. Add the pixel noise
    pixel_noise = jax.random.normal(key, shape=(N,N)) * jnp.sqrt(noise_variance_field)
    unmasked_obs = delta + pixel_noise

    #4. Masking the observation
    masked_obs = jnp.where(mask, 0., unmasked_obs)

    #Note: Step 3 and 4 are not necessary in explicit inference
    #as it is accounted for in the log-likelihood function using
    #the noise variance field.

    return masked_obs



# %%
# echo: True
cosmo_params = jnp.array([A_s, n_s, f_NL])
rng_key = jax.random.PRNGKey(0)

simulated_obs = simulator(cosmo_params, rng_key)

#Generate another field with the same cosmological parameters but different ICs
rng_key, _ = jax.random.split(rng_key)
simulated_obs2 = simulator(cosmo_params, rng_key)


# %%
#| label: fig-simulated_fields
#| fig-cap: "Two realisations of simulated fields with the same cosmological parameters but different initial conditions."
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 6))


im0 = ax0.imshow(np.ma.masked_where(mask, simulated_obs), vmin=-max(-simulated_obs.min(), simulated_obs.max()), vmax=max(-simulated_obs.min(), simulated_obs.max()), origin='lower', cmap=planck)
ax0.set_title("Simulated noisy density field")
divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im0, cax=cax0)

im1 = ax1.imshow(np.ma.masked_where(mask, simulated_obs2), vmin=-max(-simulated_obs.min(), simulated_obs.max()), vmax=max(-simulated_obs.min(), simulated_obs.max()), origin='lower', cmap=planck)
ax1.set_title("Another simulated noisy density field")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im1, cax=cax1)

plt.show()


# %% [markdown]

"""
A thing that appears is that an extra step of compression is necessary to build a summary statistics encapsulating the statistical information in the field.
Indeed, we clearly see that different ICs will produce different field but what matters is the statistics. There are different ways to perform this compression in practice.

We now generate a dataset of cosmological parameters and simulated field $(\boldsumbol{\theta}, d_\mathrm{sim})$.
"""

# %%
n_simulations = 20_000
master_key = jax.random.PRNGKey(42)
rng_keys = jax.random.split(master_key, num=n_simulations)

mu_cosmo_params = jnp.array([mu_A_s, mu_n_s, mu_f_NL])
sigma_cosmo_params = jnp.array([sigma_A_s, sigma_n_s, sigma_f_NL])

#Draw from the prior
cosmo_params_prior = jax.random.normal(master_key, shape=(n_simulations, 3)) * sigma_cosmo_params + mu_cosmo_params

# %%
#| label: fig-training-samples
#| fig-cap: Localisation of the training samples used to train the neural network.
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

A_s_samples, n_s_samples, f_NL_samples = cosmo_params_prior.T
n_thin_scatter = 10

ax0 = fig.add_subplot(2, 2, 1)

nContourLevels = 3
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
        [],
        [],
        color="black",
        marker="o",
        linestyle="None",
        markersize=2,
        label="Training Samples",
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

nContourLevels = 3
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
ax1.scatter(A_s_samples[::n_thin_scatter], f_NL_samples[::n_thin_scatter], color="black", s=2)

ax1.axhline(f_NL, color="black", ls="--", lw=1)
ax1.axvline(A_s, color="black", ls="--", lw=1)
ax1.set_xlabel("$A_\\mathrm{s}$")
ax1.set_ylabel("$f_\\mathrm{NL}$")

ax2 = fig.add_subplot(2, 2, 4)

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
ax2.scatter(n_s_samples[::n_thin_scatter], f_NL_samples[::n_thin_scatter], color="black", s=2)

ax2.axhline(f_NL, color="black", ls="--", lw=1)
ax2.axvline(n_s, color="black", ls="--", lw=1)
ax2.set_xlabel("$n_\\mathrm{s}$")
ax2.set_ylabel("$f_\\mathrm{NL}$")


plt.show()

# %%
simulations = jax.vmap(simulator)(cosmo_params_prior, rng_keys)

# %% [markdown]
"""
Let's first run a Neural Posterior Estimation (NPE) without compression to have a first idea of the result and performance of the method.
"""

# %%
#| echo: true

#First create an inference object for NPE
inference = NPE()

#Flatten the field to use it
simulations_flat = simulations.reshape((n_simulations, -1))

#Append the simulation to the inference object
inference = inference.append_simulations(cosmo_params_prior, simulations_flat)

# %% [markdown]

"""
The inference object splits the dataset in training, validation and test set to assess the convergence of the model. We can now train our model.
By default, `jaxili` will train using a Masked Autoregressive Flow (MAF) which is an expressive type of NF. Other options are available or implementable.
"""

# %%
#| echo: True
#Specify a checkpoint to save the weights of the neural network
CHECKPOINT_PATH = "."
#Turn it into an absolut path
CHECKPOINT_PATH = os.path.abspath(CHECKPOINT_PATH)

num_epochs = 500 #Number of times the network will go through the whole dataset

metrics, density_network = inference.train(
    checkpoint_path=CHECKPOINT_PATH,
    num_epochs=num_epochs
)

# %%