---
editor_options: 
  markdown: 
    wrap: 72
---

# README file

# Overview

This repository contains R scripts and functions for simulation studies
and empirical applications of the **Hidden Markov Quantile Graphical
Model (HMQGM)** and related baselines (e.g., HMGlasso).\
The code has been developed for the paper **“Hidden Markov Quantile
Graphical Model”** by B. Foroni, L. Merlo, L. Petrella, N. Salvati
(2025). The code supports the analyses reported in the manuscript (main
text §§4.1–4.3, SM §§S2.1–S2.3) and the real data application on
P`M<sub>`{=html}2.5</sub> concentrations in Northern Italy (main text
§§5).

------------------------------------------------------------------------

# Scripts

## Simulation Studies

-   **`simulations_4.1-2_HMQGM.R`**\
    Simulation study for §§4.1–4.2 fitting HMQGM. Focus: edge recovery
    (ROC curves) and clustering performance (ARI).

-   **`simulations_4.1-2_HMGlasso.R`**\
    Same design as above, but fitting HMGlasso instead of HMQGM.

-   **`simulations_4.3_HMQGM.R`**\
    Simulation study for §4.3 fitting HMQGM. Focus: performance of
    AIC/BIC/ICL for selecting the true number of hidden states.

-   **`simulations_4.3_HMGlasso.R`**\
    Same design as above, but fitting HMGlasso.

-   **`simulations_S2.1_HMQGM.R`**\
    SM §S2.1: Edge recovery fitting HMQGM when data are generated as in
    Chun et al. (2016) with K=1 state.

-   **`simulations_S2.1_HMGlasso.R`**\
    Same as above but for HMGlasso.

-   **`simulations_S2.3.R`**\
    SM §S2.3: Sensitivity analysis with smoothly time-varying adjacency
    matrices, comparing models for K=1,…,5.

## Empirical Application

-   **`RealdataScript_HMQGM.R`**\
    Application to P`M<sub>`{=html}2.5</sub> concentrations in 14 Northern Italian
    cities (2019–2022). Fits HMQGM with K=1,…,4 and saves results for
    AIC/BIC/ICL model selection, adjacency structures, selected lambdas,
    and runtime information for downstream tables/figures.

------------------------------------------------------------------------

# Functions

-   **`em_glasso.R`**\
    Implements the EM algorithm for HMGlasso (`em.glasso`).

-   **`EM_lqgm.Mix_c.R`**\
    Implements the penalized EM algorithm for HMQGM (`em.hmm.pen.lqgm`).

-   **`MainFunctions.R`**\
    Collection of helper functions for data generation and evaluation:

    -   `Theta_gen`: build precision matrices for Scenario 1.\
    -   `Ygen_Chun`: data generator as in Chun et al. (2016).\
    -   `Ygen1_sep`: conditional-quantile data generator for Scenario
        2.\
    -   `Ygen1_dynamic`: dynamic data generator with smooth
        transitions.\
    -   `Graph.performance`: compute TPR/FPR and related graph metrics.\
    -   `Viterbi`: classical Viterbi algorithm for HMM state decoding.

-   **`RealdataFun.R`**\
    Function `realdata` to fit HMQGM on real datasets with multi-start
    and lambda grid, returning selected models, criteria, adjacency
    structures, and timing.

------------------------------------------------------------------------

# Notes

-   All scripts are designed to be modular: simulation files source the
    required functions before running.\
-   Parallel execution is supported via `parallel` and `doParallel`.\
-   Non-CRAN dependencies (`rqPen`, `LQGM`) must be installed from local
    tarballs.\
-   Each simulation script saves results as `.RData` objects (`out_sim`,
    `out_glassoHM`, `results`, or `out_hmqgm_dyn`).\
-   The real data script saves one `.RData` file per K
    (`HMLQGMrealdata_<K>K.RData`).

------------------------------------------------------------------------
