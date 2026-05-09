# COMMET solve

**COMMET solve** is an open-source finite element (FE) solver for efficient deployment of neural constitutive models (NCMs) in computational mechanics simulations.
It is part of **COMMET**: the **Computational Mechanics and Machine Learning Toolbox**.

COMMET solve is currently in beta.

## Overview

Neural constitutive models are often trained in machine-learning frameworks such as PyTorch, but deploying them efficiently inside FE simulations can be difficult. 
COMMET solve is designed to address that gap by combining a high-performance finite element backend with a user-friendly input interface and optimized constitutive-model evaluation.

Key features include:

- High-performance C++ backend
- User-friendly JSON frontend
- Integration of PyTorch-based material models
- Batch-vectorized constitutive updates for fast system assembly
- Distributed-memory scalability for HPC workflows using MPI
- Ready-to-use Docker image for straightforward installation

## Links

- [COMMET project website](https://commet-code.github.io/)
- [COMMET solve documentation](https://commet-code.github.io/commet_solve/)
- [Docker image](https://hub.docker.com/repository/docker/commetcode/commet_solve)
- [GitHub repository](https://github.com/COMMET-code/commet_solve)
- [COMMET solve paper](https://doi.org/10.1016/j.cma.2026.118728)
- [Zenodo record](https://zenodo.org/records/17310682)

## Installation

For most users running COMMET solve on a laptop or workstation, the recommended installation method is Docker.

```bash
docker pull commetcode/commet_solve
