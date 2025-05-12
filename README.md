# Uncertainty for SVBRDF Acquisition using Frequency Analysis
Code release for SIGGRAPH 2025 paper "Uncertainty for SVBRDF Acquisition using Frequency Analysis"

<p>
    <a href="https://opensource.org/license/MIT"><img src="https://img.shields.io/badge/license-MIT-yellow?style=plastic" alt="license"></a>
    <a href="https://svbrdf-uncertainty.github.io/"><img src="https://img.shields.io/badge/website-Gitpage-blue?style=plastic&logo=github&logoColor=white" alt="website"></a>
    <a href="https://arxiv.org/abs/2406.17774" alt ="arXiv"> <img src="https://img.shields.io/badge/arXiv-2406.17774-b31b1b.svg"/></a>
    <a href="https://doi.org/10.1145/3721238.3730592" alt="doi">
    <img src="https://img.shields.io/badge/DOI-10.1145/3721238.3730592-red?style=plastic&logo=doi&color=red" alt="website"></a>
</p>

**Stay tuned:** Code is planned for release before June 1st.

<img src="https://svbrdf-uncertainty.github.io/static/images/teaser.jpg" alt="Overview image." />
<small>We quantify uncertainty for SVBRDF acquisition from multi-view captures (left) using entropy (right). We significantly accelerate the, otherwise heavy, computation in the frequency domain (Spherical Harmonics), yielding a practical, efficient method.</small>

## Abstract
This paper aims to quantify uncertainty for SVBRDF acquisition in multi-view captures. Under uncontrolled
illumination and unstructured viewpoints, there is no guarantee that the observations contain enough
information to reconstruct the appearance properties of a captured object. We study this ambiguity, or
uncertainty, using entropy and accelerate the analysis by using the frequency domain, rather than the
domain of incoming and outgoing viewing angles. The result is a method that computes a map of uncertainty
over an entire object within a millisecond. We find that the frequency model allows us to recover SVBRDF
parameters with competitive performance, that the accelerated entropy computation matches results with a
physically-based path tracer, and that there is a positive correlation between error and uncertainty. We
then show that the uncertainty map can be applied to improve SVBRDF acquisition using capture guidance,
sharing information on the surface, and using a diffusion model to inpaint uncertain regions.