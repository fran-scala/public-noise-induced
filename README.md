# Noise-induced equalization in quantum learning models

[![Python Version](https://img.shields.io/badge/python-3.9.6-blue.svg)](https://www.python.org/downloads/release/python-396/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-arXiv%3A2511.09428-yellow.svg)](https://doi.org/10.48550/arXiv.2511.09428)


## Overview

This repository contains the code and notebooks accompanying the paper:

**Noise-induced equalization in quantum learning models**  
<u>Francesco Scala¹</u>, Giacomo Guarnieri², and Aurelien Lucchi¹  
¹ Department of Mathematics and Computer Science, University of Basel (Switzerland)  
² Dipartimento di Fisica "A. Volta," Università di Pavia, via Bassi 6, 27100 Pavia (Italy)

### Abstract
Quantum noise is known to strongly affect quantum computation, thus potentially limiting the performance of currently available quantum processing units. 
Even learning models based on variational quantum algorithms, which were designed to cope with the limitations of state-of-the art noisy hardware capabilities, are affected by noise-induced barren plateaus, arising when the noise level becomes too strong. 
However, the generalization performances of such quantum machine learning algorithms can also be positively influenced by a proper level of noise, despite its generally detrimental effects. 

Here, we propose a pre-training procedure to determine the quantum noise level leading to desirable optimization landscape properties. 
We show that an optimized level of quantum noise induces an ‘equalization’ of the directions in the Riemannian manifold, flattening(/enhancing) the initially steep(/shallow) ones by redistributing sensitivity across its principal eigen-directions. 
We analyze this noise-induced equalization through the lens of the quantum fisher information matrix, thus providing a recipe that allows to estimate the noise level inducing the strongest equalization. 
We finally benchmark these conclusions with extensive numerical simulations providing evidence of the beneficial noise effects in the neighborhood of the best equalization, often leading to improved generalization.

---

## Repository Structure

| File | Description |
|------|--------------|
| `compare_opt_p.py` | Comparison of optimal noise levels across different datasets and models. |
| `datasets.py` | Functions for generating datasets used in experiments. |
| `generalization_bound.ipynb` | Analysis of generalization bound behavior under varying noise conditions. |
| `NIE_tools.py` | Helper functions for analysis done in `noise_induced_equalization.ipynb`. |
| `noise_induced_equalization.ipynb` | Main notebook demonstrating the Noise-Induced Equalization effect. |
| `training_models.ipynb` | Training QNNs and comparing the optiml noise from NIE with the one giving best generalization. |
| `requirements.txt` | Python dependencies for reproducing results. |

---

## Installation

This project was developed with **Python 3.9.6**.  
Some libraries may use slightly older versions than current releases.

1. Clone the repository:
   ```
   bash
   git clone <repository_url>
   cd <repository_folder>```

2. Create and activate a virtual environment:
    ```
    python -m venv myenv
    source myenv/bin/activate        # On Linux or macOS
    myenv\Scripts\activate           # On Windows```
    
3. Install dependencies:
    ```pip install -r requirements.txt```

## Usage

`noise_induced_equalization.ipynb` should be the first file to be executed, since it creates QFIMs that will be used by other files as well.

Then `training_models.ipynb` and `generalization_bound.ipynb` can be executed. 

Lastly `compare_opt_p.py`.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
