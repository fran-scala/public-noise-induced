# Improving Quantum Neural Networks Exploration by Noise-Induced Equalization

[![Python Version](https://img.shields.io/badge/python-3.9.6-blue.svg)](https://www.python.org/downloads/release/python-396/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-to_be_added-yellow.svg)](#)

## Overview

This repository contains the code and notebooks accompanying the paper:

**Improving Quantum Neural Networks Exploration by Noise-Induced Equalization**  
<u>Francesco Scala¹</u>, Giacomo Guarnieri², and Aurelien Lucchi¹  
¹ Department of Mathematics and Computer Science, University of Basel (Switzerland)  
² Dipartimento di Fisica "A. Volta," Università di Pavia, via Bassi 6, 27100 Pavia (Italy)

### Abstract
Quantum noise can strongly affect quantum computation, limiting the performance of current quantum hardware.  
Even variational quantum algorithms and quantum neural networks (QNNs) experience *noise-induced barren plateaus* at high noise levels.  
However, a moderate amount of noise can improve generalization performance.

This work proposes a pre-training procedure to determine the optimal quantum noise level that enhances the optimization landscape.  
We show that an appropriate noise level induces a **Noise-Induced Equalization** of the variational parameters, analyzed through the **Quantum Fisher Information Matrix (QFIM)**.  
Numerical simulations confirm that the region around this optimal equalization often yields improved generalization.

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