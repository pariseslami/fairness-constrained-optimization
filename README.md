# As Fair as Possible: Constrained Optimization of Fairness under Accuracy Bounds

**ICDM 2026**

## Abstract

Fairness-aware classification methods typically treat predictive 
accuracy as the primary objective and enforce fairness as a 
penalty or constraint, an approach that offers no direct guarantee 
on how much accuracy may degrade. We propose a complementary 
formulation that inverts this priority: fairness is the objective 
and accuracy is the constraint, enforced through a user-specified 
performance budget that bounds the allowable degradation relative 
to an unconstrained baseline. Under this formulation, any feasible 
solution carries a provable accuracy guarantee, while the optimizer 
is free to minimize unfairness as aggressively as the constraint 
permits. We study five gradient-based methods for solving this 
constrained problem on neural network classifiers, including our 
two proposed methods: **LBALM** (Log-Barrier Augmented Lagrangian), 
which combines the barrier method with the augmented Lagrangian 
dual update, and **AFRIAL** (Adaptive Feasibility-Restoring 
Interior-point Augmented Lagrangian), which extends LBALM by 
replacing the fixed barrier parameter with an adaptive schedule 
that responds to the constraint status at each epoch. Experiments 
on eight benchmark datasets across 16 settings show that LBALM 
achieves 100% feasibility in 10 out of 16 settings and improves 
over all three baselines in every metric. AFRIAL further achieves 
the highest average feasibility (97.8%) and fairness improvement 
(+0.207 in p%-rule), and is the only method that never completely 
fails across any setting.

We study five gradient-based methods under this formulation:

| Method | Description |
|--------|-------------|
| LB | Log-Barrier interior-point method |
| PD | Primal-Dual (Lagrange multiplier) |
| ALM | Augmented Lagrangian Method |
| **LBALM** | Log-Barrier + ALM (Proposed I) |
| **AFRIAL** | Adaptive Feasibility-Restoring Interior-point Augmented Lagrangian (Proposed II) |

LBALM achieves 85.2% average feasibility versus 54.2% for 
ALM. AFRIAL further raises this to 97.8% while achieving 
the highest average p%-rule improvement (+0.207), never 
completely failing on any setting.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

We evaluate on eight benchmark datasets:

- **Adult Income** — [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- **COMPAS Recidivism** — [ProPublica](https://github.com/propublica/compas-analysis)
- **German Credit** — [UCI Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Bank Marketing** — [UCI Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Law School GPA** — [SEAPHE](http://www.seaphe.org/databases.php)
- **MEPS 19, 20, 21** — loaded via [AIF360](https://github.com/Trusted-AI/AIF360)

See `datasets/README.md` for detailed download and 
preprocessing instructions.

## Running Experiments

```bash
# Run a single method on a single dataset
python experiments/run_adult.py --method afrial --attr sex
python experiments/run_compas.py --method lbalm --attr race

# Run all methods on all datasets
python experiments/run_all.py
```

## Results

Average results across 16 dataset-attribute settings:

| Method | Avg Feasibility | Avg Δp%-rule | Avg Δacc |
|--------|----------------|--------------|----------|
| LB     | 75.4%          | +0.166       | -0.029   |
| PD     | 56.0%          | +0.100       | -0.031   |
| ALM    | 54.2%          | +0.158       | -0.010   |
| **LBALM**  | **85.2%**  | **+0.202**   | -0.010   |
| **AFRIAL** | **97.8%**  | **+0.207**   | -0.011   |

## Repository Structure

@inproceedings{author2026fairness,
  title     = {As Fair as Possible: Constrained Optimization 
               of Fairness under Accuracy Bounds},
  booktitle = {Proceedings of the IEEE International Conference 
               on Data Mining (ICDM)},
  year      = {2026}
}
