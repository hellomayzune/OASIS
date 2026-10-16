<p align="left">
• <a href="https://github.com/hellomayzune"><strong>GitHub</strong></a> •
<a href="https://orcid.org/0000-0003-0282-2633"><strong>ORCID</strong></a> •
<a href="https://scholar.google.com/citations?user=LmP8B_4AAAAJ&hl=en"><strong>Google Scholar</strong></a> •
<a href="https://www.researchgate.net/profile/May-Zune"><strong>ResearchGate</strong></a> •
<a href="https://www.linkedin.com/in/mayzune/"><strong>Linkedin</strong></a> •
<a href="https://mayzune.com/"><strong>Blog</strong></a> •

</p>

# 🌡️ OASIS: Overheating Assessment through Screening-Level Intervention Scenarios

This repository houses the Jupyter notebooks, input data descriptions, and processed outputs for **OASIS**, a methodological case study dedicated to residential overheating assessment. OASIS rigorously segregates three distinct streams of evidence to evaluate domestic thermal performance.

The first stream comprises room-level empirical monitoring spanning four consecutive cooling seasons (2023–2026), assessed directly against CIBSE TM59 benchmarks. The second stream consists of a dynamic thermal model developed in EnergyPlus and checked against measurements over an unoccupied period. The third stream delivers room-level machine-learning surrogates benchmarked against persistence, linear, and grey-box thermal baselines. These surrogates are tested recursively and under extreme heat, and applied to DSY2030 and DSY2080 future weather.

- **Project**: OASIS — Overheating Assessment through Screening-Level Intervention Scenarios
- **Authors**: [Renata Tubelo](https://orcid.org/0000-0001-9607-7616), [May Zune](https://orcid.org/0000-0003-0282-2633), [Julie Waldron](https://orcid.org/0000-0002-7572-1326)
- **Notebook author**: May Zune
- **Funding**: This research received no specific grant from funding agencies. The authors declare no competing interests.
- **Associated publication**: manuscript in preparation; citation and Zenodo DOI to be added.

📬 Contact: please open an issue, or use the links above.

🆚 Version: v1

©️ Licence: code under the *MIT Licence*; the authors' processed data and outputs under *CC BY 4.0*. The raw monitoring record and third-party weather data are not redistributed (see [What is and isn't included](#-what-is-and-isnt-included)).

📖 Contents

- [Research questions](#-research-questions)
- [Repository information](#️-repository-information)
- [Notebooks](#️-notebooks)
- [Software requirements & setup](#-software-requirements--setup)
- [Reproducibility steps](#-reproducibility-steps)
- [Project documentation](#-project-documentation)
- [What is and isn't included](#-what-is-and-isnt-included)

---

# ❓ Research questions

| | Question | Notebooks |
|---|---|---|
| **RQ1** | How do monitored indoor temperatures compare with the CIBSE TM59 criteria over the monitored cooling seasons? | Stage 2 (with the Stage 1 outdoor-record check) |
| **RQ2** | Do machine-learning surrogates add predictive information beyond persistence and simple thermal baselines (a linear ARX model and a 1R1C grey-box model), under recursive forecasting and extreme heat, and does the prediction target matter? | Stages 4 and 5 |
| **RQ3** | Can the physics model and the benchmarked surrogates support future-climate screening, and how far do dwelling- and room-level conclusions depend on model form and uncertainty? | Stages 3A, 3B, 5 and 6 |

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# 🗂️ Repository information

- Contents: Jupyter notebooks (`.ipynb`) with HTML previews, input files (`.xlsx`, `.csv`), and generated tables, figures and model files (`.csv`, `.png`, `.pkl`, `.json`)
- Programming language: `Python` (`pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `xgboost`, `shap`, `scikit-learn`, `joblib`, `openpyxl`)
- Building-physics simulation software: EnergyPlus / DesignBuilder (produces the measured-vs-simulated and scenario workbooks; the simulation itself is not re-run from this repository)
- README created: 26 September 2026
- README modified: 15 October 2026

📌 Repository map

```
├── README.md                                        # Project overview (this file)
├── DATASHEET.md                                     # Dataset documentation (Gebru et al., 2021)
├── ADR.md                                           # Architecture decision records
├── MODEL_CARD.md                                    # Model cards (Mitchell et al., 2019)
├── 1_input/
│   ├── NottinghamMeasured*.xlsx                     # Hourly room-level and outdoor monitoring record (8 sheets; NOT distributed)
│   ├── Nottingham_July2024_Validation.xlsx          # Measured vs simulated temperatures, unoccupied period, 10–30 July 2024
│   ├── Nottingham_Simulated_Temperature.xlsx        # EnergyPlus DSY2030 / DSY2080 scenario and orientation output
│   ├── 2030_Nottingham_a1fi_50p_DSY-hours.csv       # DSY2030 hourly weather (third-party, COLBE; NOT distributed)
│   ├── Hour_50th_2080s_4600345_DSY.csv              # DSY2080 hourly weather (third-party, COLBE; NOT distributed)
│   └── midas-open_uk-hourly-weather-obs_*.csv       # Met Office MIDAS Open hourly observations, Sutton Bonington, 2023–2025 (third-party; NOT distributed)
├── stage1-oasis-vs-midas-outdoor-comparison.ipynb
├── stage2-empirical-overheating-assessment.ipynb
├── stage3a-model-validation.ipynb
├── stage3b-simulated-temp-future-scenario-analysis.ipynb
├── stage4-ml-shap-pipeline.ipynb
├── stage5-exceedance-cdh-pipeline.ipynb
├── stage6-ml-ai-advanced-synthesis.ipynb
├── stage*-preview.html                              # HTML export of each executed notebook (last cell)
└── 2_output/                                        # Stage 1–3B tables and figures
    ├── figures/                                     # Stage 4–6 figures
    ├── tables/                                      # Stage 4–6 tables
    └── models/                                      # Stage 4 evaluation and Stage 5 production surrogates (.pkl + .json)
```

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# ⚙️ Notebooks

The repository's Jupyter notebooks are structured around the multi-stage OASIS framework. Stage 1 covers data collection and preprocessing, including the check of the on-site outdoor record, which feeds into the empirical overheating assessment in Stage 2. Stage 3 handles physics-based model checking along with future-climate scenario simulations. Stage 4 introduces the machine-learning surrogates, conducting baseline benchmarking and SHAP interpretability analysis. Stage 5 evaluates future-weather thermal exposure and performs cooling degree-hour screening. Stage 6 integrates reliability checks, room-level synthesis, and overall screening to conclude the pipeline.

### 📊 Stage 1 — Outdoor record check against MIDAS (RQ1, supporting)
*See `DATASHEET.md` Datasets A and E, `ADR-046`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| Compares the on-site outdoor logger with Met Office MIDAS Open observations at Sutton Bonington: coverage and quality flags, time-alignment check, hourly, daily and monthly statistics with block-bootstrap intervals, running-mean (Trm) comparison with a like-for-like initialisation, the resulting shift in the EN 16798-1 and ASHRAE 55 adaptive limits and comfort-hour counts, and rolling ΔTrm | `NottinghamMeasuredOASIS.xlsx` (External sheet and five indoor rooms), `midas-open_uk-hourly-weather-obs_*.csv` | Statistics and summary in the notebook; `measured_vs_midas_temperature.png`, `measured_vs_midas_rh_dew.png` |

### 📊 Stage 2 — Empirical overheating assessment (RQ1)
*See `DATASHEET.md` Dataset A, `MODEL_CARD.md` Model 1, `ADR-006`–`ADR-012`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| Merges the monitoring workbooks; computes the running-mean outdoor temperature; applies TM59:2017 Criterion (a) and TM59:2026 Criteria (b) and (d) by room and season; secondary Heat Index, stratification, room correlation, missing-data and night-time cooling diagnostics; steady-state estimate of the operative-minus-air temperature offset and its effect on the TM59 outcomes | `NottinghamMeasured*.xlsx` | `stage2_TM59_Criterion_a/b/d_results.csv`, `stage2_operative_vs_air_estimate.csv`, `stage2_operative_offset_sensitivity.csv`, `stage2_HeatIndex_summary.csv`, `stage2_missing_data_audit.csv`, `stage2_nighttime_decay_rates.csv`, figures |

### 📊 Stage 3A — Physics-model check against measurements (RQ3)
*See `DATASHEET.md` Dataset B, `MODEL_CARD.md` Model 2, `ADR-013`–`ADR-017`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| Statistical check of the EnergyPlus model over an unoccupied period: RMSE, Cv(RMSE), NMBE, MAE and R² per zone with block-bootstrap intervals, against CIBSE TM63 / ASHRAE Guideline 14 hourly tolerances | `Nottingham_July2024_Validation.xlsx` | `stage3a_validation_data_cleaned.csv`, `stage3a_validation_summary_table.csv`, figures |

### 📊 Stage 3B — Simulated scenarios under future weather (RQ3)
*See `DATASHEET.md` Datasets C–D, `MODEL_CARD.md` Model 2, `ADR-018`–`ADR-020`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| TM59 Criteria (a) and (b) for the as-built case, five mitigation scenarios and four orientations under DSY2030 and DSY2080; scenario efficacy and climate escalation | `Nottingham_Simulated_Temperature.xlsx`, DSY weather CSVs | `stage3b_TM59_Criterion_a_scenarios.csv`, `stage3b_TM59_Criterion_a/b_orientation.csv`, `stage3b_TM59_by_horizon.csv`, `stage3b_scenario_bedroom/living_pct_reduction.csv`, `stage3b_table_climate_escalation_summary.csv`, figures |

### 📊 Stage 4 — ML surrogate, benchmarks and SHAP (RQ2; DSY projection for RQ3)
*See `MODEL_CARD.md` Models 3–5, `ADR-021`–`ADR-031`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| Room-level XGBoost one-step surrogates on a chronological split; persistence, diurnal persistence, ARX, 1R1C and weather-only benchmarks; heatwave and rolling-origin recursive forecasts with split-conformal intervals; upper-tail and outdoor-temperature-bin diagnostics; temperature-change (ΔT) XGBoost as a controlled alternative; SHAP; DSY projection and comparison with the physics model | Monitoring workbook, simulated workbook, DSY weather CSVs, Stage 3B `stage3b_TM59_by_horizon.csv` | `tables/stage4_*.csv`, `shap_*.csv`, `dsy_*.csv`, `stage4_model_manifest.csv`, `models/stage4_xgb_<Room>.pkl/.json`, `figures/stage4_*.png` |

### 📊 Stage 5 — Future-weather exposure and cooling-degree-hours (RQ2, RQ3)
*See `MODEL_CARD.md` Models 4–6, `ADR-032`–`ADR-038`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| 80/20 recursive validation against held-seed persistence, ARX and 1R1C; year-long test of ΔT-XGBoost; refit of production surrogates; DSY projection with seed sensitivity and extrapolation audit; hours above 26 °C with a split-conformal interval; cross-model Passive/Hybrid sensitivity experiments; Active temperature ceilings and CDH; exposure by model form, bias correction by outdoor temperature, and threshold sensitivity (25–28 °C) | Monitoring workbook, DSY weather CSVs, simulated workbook (Passive/Hybrid offsets) | `tables/stage5_*.csv` (including `stage5_model_manifest.csv`), `models/stage5_production_xgb_<Room>.pkl/.json`, `figures/stage5_*.png` |

### 📊 Stage 6 — Reliability, room-level synthesis and screening (RQ3)
*See `MODEL_CARD.md` Model 7, `ADR-039`–`ADR-045`.*

| Purpose | Key inputs | Produces |
|---|---|---|
| Verifies the Stage 4/5 hand-off; surrogate reliability and confidence flags; uniform-warming stress test; model-instance spread; k = 2 vs k = 3 room clustering with stability checks; intervention and Active screening; DSY hazard characterisation; room-level screening matrix; tier sensitivity to model form and interval bounds | Stage 4 and 5 tables and model bundles, DSY weather CSVs | `tables/Stage6_*.csv`, `figures/Stage6_*.png` |

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# 💻 Software requirements & setup

**Physics-based simulation** (not re-run here): EnergyPlus / DesignBuilder, driven by the measured outdoor record (check period) and by the COLBE DSY2030 and DSY2080 (50th percentile) weather files (scenarios).

**Notebook environment:**
- Python 3.10 or later, Jupyter Notebook or JupyterLab (last tested with Python 3.13, pandas 3.0, xgboost 3.4, shap 0.52, scikit-learn 1.9)
- Packages: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `openpyxl`, `xgboost`, `shap`, `scikit-learn`, `joblib`, `nbconvert` (HTML preview export in the last cell of each notebook)

```bash
conda create -n oasis-env python=3.11
conda activate oasis-env
conda install -c conda-forge pandas numpy scipy matplotlib seaborn openpyxl xgboost shap scikit-learn joblib jupyter nbconvert
```

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# 🔁 Reproducibility steps

Place the input files listed in the repository map in `1_input/`. Each notebook reads only from `1_input/` (and, from Stage 4 onward, from earlier outputs in `2_output/`) and writes only to `2_output/`; input files are never modified (`ADR-002`). Room names, monitoring years, train/test years and the heatwave window are derived from the data, so a newer monitoring workbook needs no code change (`ADR-005`).

Run order:

1. `stage1-oasis-vs-midas-outdoor-comparison.ipynb` (independent of the other stages)
2. `stage2-empirical-overheating-assessment.ipynb`
3. `stage3a-model-validation.ipynb`
4. `stage3b-simulated-temp-future-scenario-analysis.ipynb` (writes `stage3b_TM59_by_horizon.csv`, read by Stage 4)
5. `stage4-ml-shap-pipeline.ipynb` (writes the Stage 4 model bundles read by Stage 6)
6. `stage5-exceedance-cdh-pipeline.ipynb` (writes the production model bundles read by Stage 6)
7. `stage6-ml-ai-advanced-synthesis.ipynb` (no re-training; verifies that the reloaded models reproduce the Stage 5 results, `ADR-039`)

**What can be re-run from the public files.** The raw monitoring record is not distributed, and the DSY and MIDAS files must be downloaded from their providers (see below). With the downloaded weather files:

| Notebook | Re-runnable without the monitoring record? |
|---|---|
| Stage 1 | No (needs the External and indoor sheets) |
| Stage 2 | No; results are in the saved `2_output/stage2_*` tables |
| Stage 3A | Yes |
| Stage 3B | Yes (needs the DSY CSVs) |
| Stages 4 and 5 | No; results, model bundles and manifests are in `2_output/` |
| Stage 6 | Yes, from the saved Stage 4–5 tables and model bundles (needs the DSY CSVs) |

Stochastic steps use a fixed seed (`RANDOM_STATE = 42` / `SEED = 42`). Reported values are generated from the saved tables of a clean-kernel run, not typed into the notebooks (`ADR-003`).

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# 📖 Project documentation

| Document | Framework | Covers |
|---|---|---|
| [`DATASHEET.md`](./DATASHEET.md) | Gebru et al. (2021) | Motivation, composition, collection, preprocessing, uses, distribution and maintenance for the five input datasets (monitoring record, model-check workbook, scenario workbook, DSY weather files, MIDAS observations) and the derived outputs |
| [`ADR.md`](./ADR.md) | Lightweight MADR | The design decisions behind each stage, with the alternatives considered and their consequences |
| [`MODEL_CARD.md`](./MODEL_CARD.md) | Mitchell et al. (2019) | Intended use, factors, metrics, data and caveats for the seven models: TM59 classifier, EnergyPlus model, one-step surrogate with baselines, ΔT-XGBoost, recursive DSY projection, scenario layer, and room clustering and screening tiers |

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# 📦 What is and isn't included

**Included:**
- The notebooks and their HTML previews (MIT Licence).
- The two EnergyPlus workbooks (`Nottingham_July2024_Validation.xlsx`, `Nottingham_Simulated_Temperature.xlsx`) and every file in `2_output/` (CC BY 4.0).

**Not included:**
- **Raw monitoring record** (`NottinghamMeasured*.xlsx`): not publicly available because of confidentiality agreements and the occupants' privacy. Its structure is documented in `DATASHEET.md` (Dataset A) so the notebooks can be run on a dataset of the same form.
- **DSY weather files** (`2030_Nottingham_a1fi_50p_DSY-hours.csv`, `Hour_50th_2080s_4600345_DSY.csv`): Design Summer Year files from COLBE (University of Bath), available from [colbe.bath.ac.uk](https://colbe.bath.ac.uk/) subject to COLBE's terms of use.
- **MIDAS observations** (`midas-open_uk-hourly-weather-obs_*.csv`): Met Office MIDAS Open hourly observations (station 00554, Sutton Bonington), available from the Centre for Environmental Data Analysis ([archive.ceda.ac.uk](https://archive.ceda.ac.uk/)) under the Open Government Licence, which requires attribution.
- The EnergyPlus / DesignBuilder project files and the simulation runs themselves (see `ADR-013`), and any information identifying the dwelling's address or occupants.

[Back to top](#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)
