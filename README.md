<h1 id="top">Uncertainty for SVBRDF Acquisition using Frequency Analysis</h1>
Code release for SIGGRAPH 2025 paper **"Uncertainty for SVBRDF Acquisition using Frequency Analysis"**<br />
<a href="https://rubenwiersma.nl/" target="_blank">Ruben Wiersma</a>, 
<a href="https://julienphilip.com/" target="_blank">Julien Philip</a>, 
<a href="https://miloshasan.net/" target="_blank">Milos Hasan</a>,
<a href="https://krishnamullia.com/" target="_blank">Krishna Mullia</a>, 
<a href="https://luanfujun.com/" target="_blank">Fujun Luan</a>, 
<a href="https://graphics.tudelft.nl/" target="_blank">Elmar Eisemann</a>,
<a href="https://valentin.deschaintre.fr" target="_blank">Valentin Deschaintre</a><br />

<p>
    <a href="https://opensource.org/license/MIT"><img src="https://img.shields.io/badge/license-MIT-yellow?style=plastic" alt="license"></a>
    <a href="https://svbrdf-uncertainty.github.io/"><img src="https://img.shields.io/badge/website-Gitpage-blue?style=plastic&logo=github&logoColor=white" alt="website"></a>
    <a href="https://arxiv.org/abs/2406.17774" alt ="arXiv"> <img src="https://img.shields.io/badge/arXiv-2406.17774-b31b1b.svg"/></a>
    <a href="https://doi.org/10.1145/3721238.3730592" alt="doi">
    <img src="https://img.shields.io/badge/DOI-10.1145/3721238.3730592-red?style=plastic&logo=doi&color=red" alt="website"></a>
</p>

<a href="#setup">Setup</a> • <a href="#quickstart">Quickstart</a> • <a href="#instructions">Instructions</a> • <a href="#license">License</a> • <a href="#roadmap">Roadmap</a> • <a href="#citation">Citation</a>

<img src="https://svbrdf-uncertainty.github.io/static/images/teaser.jpg" alt="Overview image." />
<small>We quantify uncertainty for SVBRDF acquisition from multi-view captures (left) using entropy (right). We significantly accelerate the, otherwise heavy, computation in the frequency domain (Spherical Harmonics), yielding a practical, efficient method.</small>

<h2 id="setup">🗝 Setup</h2>
<p align="right"><a href="#top"><sup>▴ Back to top</sup></a></p>

### Environment

To run our code, you need to setup a (conda) environment and install the `svbrdf_uncertainty` package.
First, clone the repository:
```bash
git clone https://github.com/rubenwiersma/svbrdf_uncertainty.git
```

You can run a bash script we prepared to setup the environment, named `svbrdf_uncertainty`:
```bash
cd svbrdf_uncertainty
source setup.sh
```
This will also install the required dependencies, such as Mitsuba, nvdiffrast, and torch.

Make sure the environment is activated when you run the experiments.
```bash
conda activate svbrdf_uncertainty
```

<h2 id="quickstart">🚀 Quickstart</h2>
<p align="right"><a href="#top"><sup>▴ Back to top</sup></a></p>

To check if the setup was done correctly, you can run our optimization scripts for the spherical harmonics model and Mitsuba on the plane toy example:

```bash
python opt_sh.py # Runs our method
python opt_mi.py # Runs Mitsuba
```
This runs the optimization and entropy computation for our method. The results will be placed in the folder `out/plane/[sh/mitsuba]/default`.

The results folders contain `.exr` files with the output textures (`base_color.exr`, `entropy.exr` - uncertainty, `metallic.exr`, `roughness.exr`) and an overview figure `brdf_overview.png` with these textures and re-renders.

<h2 id="instructions">🪄 Instructions</h2>
<p align="right"><a href="#top"><sup>▴ Back to top</sup></a></p>

To run the experiments in our paper, specifically for Stanford ORB, you need to download Stanford ORB and run the corresponding scripts.

Skip to:<br />
<a href="stanford-orb-data">Data</a> • <a href="#running-the-experiments">Running experiments</a> • <a href="#configurations">Configurations</a>

### Stanford ORB Data
Download and uncompress the files for Stanford ORB using the following script:
```
source stanford_orb_download.sh
```
This script will create a new folder outside of the repository in `../data/Stanford ORB` and download Stanford ORB to this folder.

Finally, run the following python script to correct the normals in the `.obj` files given by Stanford ORB (their y and z components are swapped).
```
conda activate svbrdf_uncertainty
python scripts/normal_correction_stanford_orb.py
```
**Note:** This script and many other locations in the code assume that you downloaded Stanford ORB to `../data/Stanford ORB`.

### Running the experiments
You can use the following scripts to run the Stanford ORB benchmark for our approach and Mitsuba:
```bash
python scripts/opt_sh_stanford_orb.py # Our method
python scripts/opt_mi_stanford_orb.py # Mitsuba
```
These scripts estimate material textures for each scene in Stanford ORB, perform relighting (combined ~10-15min), and compute the metrics in the Stanford ORB benchmark (~4min). 

The benchmark results will be stored in `out/stanford_orb/sh_baseline.json` and `out/stanford_orb/mi_baseline.json`. You can format the results for easier readibility to LaTeX using the script `scripts/json_to_latex.py`:
```bash
python scripts/json_to_latex.py out/stanford_orb
```
The output is a `.tex` file in `out/stanford_orb/stanford_orb.tex`, containing the results from all `.json` files in `out/stanford_orb`. These tables correspond to the results in Table 2 of the paper.

You can also inspect the resulting textures and entropy maps in the folders for each scene, e.g., `out/stanford_orb/blocks_scene002`.

### Configurations
We make extensive use of the [gin config](https://github.com/google/gin-config) package to configure experiments. You can decorate a function with `@gin.configurable` to configure it with `.gin` files. We use this in the following places:
- **opt_sh.py, `optimize_material()`** - The main optimization function. An example configuration script is provided in `experiments/scenes/plane`.
- **scripts/opt_sh_stanford_orb.py, `run_benchmark()`** - The benchmark function. We use this to run different variants of our method (e.g., power spectrum vs. angular domain) on the entire benchmark. The benchmark script takes a folder as input and will recursively look through all its sub-folders to find config files. If you want to run different variants of the benchmark, simply create a `.gin` file in a folder and pass the folder as an argument to `opt_sh_stanford_orb.py`:
```bash
python scripts/opt_sh_stanford_orb.py experiments/configs/new_folder
```

<h2 id="license">🔖 License</h2>
<p align="right"><a href="#top"><sup>▴ Back to top</sup></a></p>

This repository is licensed under the MIT License.

<h2 id="roadmap">🗺 Roadmap</h2>
<p align="right"><a href="#top"><sup>▴ Back to top</sup></a></p>

We aim to release code for the synthetic benchmark. This might take a bit longer due to licensing constraints.

<h2 id="citation">📖 Citation</h2>
<p align="right"><a href="#top"><sup>▴ Back to top</sup></a></p>

Please cite our paper if this code contributes to an academic publication:<br />

<pre>
@inproceedings{wiersma2025svbrdfuncertainty,
    author = {Wiersma, Ruben and Philip, Julien and Hašan, Miloš and Mullia, Krishna and Luan, Fujun and Eisemann, Elmar and Deschaintre, Valentin},
    title = {Uncertainty for SVBRDF Acquisition using Frequency Analysis},
    year = {2025},
    isbn = {979-8-4007-1540-2/2025/08},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3721238.3730592},
    doi = {10.1145/3721238.3730592},
    booktitle = {SIGGRAPH Conference Papers '25},
    location = {Vancouver, BC, CA},
    series = {SIGGRAPH Conference Papers '25}
}
</pre>