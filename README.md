# PMT-MOBO-X (CVAE/DDPM) Reimplementation

This repository contains the source code to re-implement our method **PMT-MOBO-X**, instantiated with:

- **CVAE-based generative solution modeling**
- **DDPM-based generative solution modeling**

The implementation corresponds to the IJCAI accepted paper:

**Amortized Multi-Objective Optimization Across Tasks with Generative Solution Modeling**  
Tingyang Wei, Jiao Liu, Abhishek Gupta, Chin Chun Ooi, Puay Siew Tan, Yew-Soon Ong

---

## Repository Overview

Main files:

- `test.py`: CLI entrypoint for running experiments.
- `methods.py`: core PMT-MOBO, CVAE-enhanced, and DDPM-enhanced optimization methods.
- `gen_models.py`: CVAE model and trainer.
- `simple_ddpm_model.py`: DDPM model and trainer.
- `problem.py`: Parametric DTLZ objective definitions.

---

## Environment Setup

From the parent directory (the directory that contains the `PEMOP` folder), create and activate a Python environment, then install dependencies. If you already generated `requirements.txt`, install with:

```bash
pip install -r PEMOP/requirements.txt
```

If not, install the core dependencies directly:

```bash
pip install torch numpy scipy matplotlib gpytorch pymoo pandas
```

---

## Easy CLI Usage
### 1) CVAE instantiation (VAE-CMOBO)

```bash
python PEMOP/test.py \
  --method_name VAE-CMOBO \
  --problem dtlz2 \
  --acquisition_type UCB \
  --n_runs 1 \
  --n_iter 5 \
  --n_objectives 2 \
  --n_variables 5 \
  --m_tasks 8 \
  --m_samples 20
```

### 2) DDPM instantiation (DDPM-CMOBO)

```bash
python PEMOP/test.py \
  --method_name DDPM-CMOBO \
  --problem dtlz2 \
  --acquisition_type UCB \
  --n_runs 1 \
  --n_iter 5 \
  --n_objectives 2 \
  --n_variables 5 \
  --m_tasks 8 \
  --m_samples 20
```

---

## Useful CLI Arguments

- `--method_name`: `VAE-CMOBO` or `DDPM-CMOBO`
- `--problem`: `dtlz1`, `dtlz2`, `dtlz3`
- `--acquisition_type`: `UCB` or `TS`
- `--n_runs`: number of independent runs
- `--n_iter`: BO iterations per run
- `--m_tasks`: number of contexts/tasks
- `--m_samples`: initial samples per task
- `--genai_training_frequency`: training frequency of generative model
- `--genai_top_p`: top-p selection ratio for generative training data
- `--genai_num_candidates`: number of generated candidates

---

## Outputs

- Intermediate data is stored under `data/` (contexts and initial design points).
- Optimization results and plots are saved under `result/<problem_name>/`.

---

## Citation

If you use this codebase, please cite:

**Amortized Multi-Objective Optimization Across Tasks with Generative Solution Modeling**  
Tingyang Wei, Jiao Liu, Abhishek Gupta, Chin Chun Ooi, Puay Siew Tan, Yew-Soon Ong

