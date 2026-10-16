<p align="left">
• <a href="https://github.com/hellomayzune"><strong>GitHub</strong></a> •
<a href="https://orcid.org/0000-0003-0282-2633"><strong>ORCID</strong></a> •
<a href="https://scholar.google.com/citations?user=LmP8B_4AAAAJ&hl=en"><strong>Google Scholar</strong></a> •
<a href="https://www.researchgate.net/profile/May-Zune"><strong>ResearchGate</strong></a> •
<a href="https://www.linkedin.com/in/mayzune/"><strong>Linkedin</strong></a> •
<a href="https://mayzune.com/"><strong>Blog</strong></a> •

</p>

<p align="left">
📄 Companion to <a href="./README.md"><strong>README.md</strong></a> •
<a href="./DATASHEET.md"><strong>DATASHEET.md</strong></a> •
<a href="./ADR.md"><strong>ADR.md</strong></a>
</p>

# 🧠 OASIS — MODEL CARD

This document provides standardized model cards, structured in accordance with the reporting framework established by **Mitchell et al. (2019), "Model Cards for Model Reporting,"** for each of the seven analytical models implemented across the **OASIS** (Overheating Assessment through Screening-Level Intervention Scenarios) notebook suite. To ensure comprehensive documentation across the complete analytical pipeline, rule-based empirical models and physics-based dynamic simulations are documented alongside machine-learning surrogates, as each component generates foundational outputs upon which downstream models rely.

These model cards systematically define model architectures, operational boundaries, measurement protocols, and primary output destinations. In accordance with architectural decision record `ADR-003`, numerical results are not quoted in this text; they are written to, and should be read from, the output files listed under *Quantitative analyses*.

📖 Contents

- [Model 1 — Empirical TM59 classifier and Heat Index](#-model-1--empirical-tm59-classifier-and-heat-index)
- [Model 2 — EnergyPlus dynamic thermal model](#-model-2--energyplus-dynamic-thermal-model)
- [Model 3 — One-step XGBoost surrogate and baselines](#-model-3--one-step-xgboost-surrogate-and-baselines)
- [Model 4 — Temperature-change (ΔT) XGBoost](#-model-4--temperature-change-δt-xgboost)
- [Model 5 — Recursive future-weather (DSY) projection](#-model-5--recursive-future-weather-dsy-projection)
- [Model 6 — Scenario layer: Passive, Hybrid and Active](#-model-6--scenario-layer-passive-hybrid-and-active)
- [Model 7 — Room clustering and screening tiers](#-model-7--room-clustering-and-screening-tiers)

[Back to README](./README.md#️-oasis-overheating-assessment-through-screening-level-intervention-scenarios)

---

# 📏 Model 1 — Empirical TM59 classifier and Heat Index

## Model details
- **Notebook:** `stage2-empirical-overheating-assessment.ipynb`.
- **Type:** deterministic, rule-based classifier applied to measured hourly temperatures; not fitted.
- **Rules:**
  - Running-mean outdoor temperature Trm = 0.8·Trm(n−1) + 0.2·Tod(n−1), computed continuously over the record and limited to 10–30 °C.
  - Category II: Tcomf = 0.33·Trm + 18.8; Tmax = Tcomf + 3 K.
  - **Criterion (a)** (TM59:2017; living rooms and kitchens 09:00–22:00, bedrooms 24 h; 1 May–30 September): an hour exceeds when ΔT = T − Tmax, rounded to the nearest whole degree, is at least 1 K; pass if no more than 3% of occupied hours exceed.
  - **Criterion (b)** (TM59:2026; bedrooms, 23:00–08:00): a night fails when the sleeping-period mean exceeds 27 °C; pass if no more than 4 nights fail; nights with fewer than 6 of 9 readings are not evaluated.
  - **Criterion (d)** (TM59:2026; Hall and Landing, 24 h): share of hours above 28 °C against a 3% limit, reported as a significant-risk flag for internal circulation spaces.
  - **Heat Index** (NWS Rothfusz equation) from temperature and RH, with exceedance at 25, 27 and 30 °C.
  - **Operative-offset sensitivity:** a steady-state surface balance brackets the operative-minus-air difference for sunlit daytime and night-time cases, and the criteria are recomputed with the offsets applied.
- **References:** CIBSE TM59 (2017, 2026); CIBSE TM52; BS EN 16798-1; NWS Heat Index equation.

## Intended use
- **Uses:** quantifying as-occupied overheating in the case-study dwelling by room and season; the measured reference against which the simulation and surrogate results are compared.
- **Users:** the study authors; researchers looking for a worked example of TM59 applied to monitoring data.
- **Out of scope:** design-stage compliance with TM59 or Approved Document O; other dwellings without checking sensor placement and the outdoor reference.

## Factors
Room (7), zone type, floor, orientation and season year. Results are always reported by room and season, never pooled.

## Metrics
Percentage of occupied hours exceeding (Criteria (a) and (d)); failure nights per season (Criterion (b)); Heat Index exceedance; missing-data share; night-time cooling rate (01:00–06:00); TM59 outcomes with the operative offset applied. Thresholds are constants in the configuration cell.

## Data
Dataset A, full record. No train/test split.

## Quantitative analyses
`stage2_TM59_Criterion_a_results.csv`, `stage2_TM59_Criterion_b_results.csv`, `stage2_TM59_Criterion_d_results.csv`, `stage2_operative_vs_air_estimate.csv`, `stage2_operative_offset_sensitivity.csv`, `stage2_HeatIndex_summary.csv`, `stage2_missing_data_audit.csv`, `stage2_nighttime_decay_rates.csv`.

## Ethical considerations
Temperature and humidity from one private dwelling; no occupant-identifying, health or demographic data.

## Caveats and recommendations
- Air temperature is used as a proxy for operative temperature; read results together with the operative-offset sensitivity.
- The outdoor logger reads warmer than the nearest Met Office station, which raises the adaptive limit slightly (see the Stage 1 MIDAS comparison notebook).
- Room-years with later sensor installation have a smaller base and are flagged.

[Back to top](#-oasis--model-card)

---

# 🏠 Model 2 — EnergyPlus dynamic thermal model

## Model details
- **Notebooks:** `stage3a-model-validation.ipynb` (check against measurements) and `stage3b-simulated-temp-future-scenario-analysis.ipynb` (scenarios).
- **Type:** physics-based dynamic thermal simulation in EnergyPlus / DesignBuilder, run outside this repository. Inputs were audited against as-built evidence: external wall U = 0.25, ceiling to loft 0.12, windows 1.57 W/m²K (SHGC 0.69); design air permeability 6.2 m³/h·m² at 50 Pa; about 45 cm eaves; temperature-dependent window operation; TM59 occupancy, lighting and equipment gains in the scenarios.
- **Scenarios:** as built; Case 1 (increased ventilation); 2A and 2B (Case 1 with internal blinds or external shutters); 3A and 3B (improved glazing with Case 1 and internal blinds or external shutters); four orientations of the as-built case. Weather: DSY2030 and DSY2080 (Dataset D).

## Intended use
- **Uses:** comparing mitigation scenarios and orientations under standardised TM59 conditions and future weather; a physics-based reference for the surrogate projections; the source of the Passive and Hybrid offsets (Model 6).
- **Out of scope:** predicting temperatures under the household's own operation; retrofit performance claims; the Hall and Landing, which were not checked.

## Factors
Zone (five checked: Kitchen, Living Room, Master Bedroom, Bedroom 4, Bedroom 3), scenario, orientation and climate horizon.

## Metrics
- **Check against measurements (Stage 3A):** RMSE and MAE in kelvin (primary), R², and Cv(RMSE) and NMBE against the CIBSE TM63 / ASHRAE Guideline 14 hourly tolerances (Cv(RMSE) < 30%, NMBE within ±10%), each with 24-h moving-block bootstrap intervals, per zone.
- **Scenarios (Stage 3B):** TM59 Criterion (a) exceedance and Criterion (b) failure nights by room, scenario, orientation and horizon; reductions relative to the as-built case.

## Data
- **Check:** Dataset B, 504 hours, 10–30 July 2024, unoccupied with windows closed, driven by the measured outdoor temperature.
- **Scenarios:** Dataset C, with outdoor temperature from Dataset D.

## Quantitative analyses
`stage3a_validation_summary_table.csv`; `stage3b_TM59_Criterion_a_scenarios.csv`, `stage3b_TM59_Criterion_a_orientation.csv`, `stage3b_TM59_Criterion_b_orientation.csv`, `stage3b_TM59_by_horizon.csv`, `stage3b_scenario_bedroom_pct_reduction.csv`, `stage3b_scenario_living_pct_reduction.csv`, `stage3b_table_climate_escalation_summary.csv`.

## Ethical considerations
As Model 1.

## Caveats and recommendations
- The model is checked against measurements over one unoccupied period; occupant gains, window operation and heatwave behaviour are untested (`ADR-014`). Hourly errors of the order of 1 K are comparable to the Criterion (a) margin, so outcomes close to a limit should be read with caution.
- Scenario results describe standardised TM59 operation and are comparisons, not predictions.
- Orientation runs keep the as-built room labels; read the room label with the rotation label (`ADR-020`).

[Back to top](#-oasis--model-card)

---

# 🌲 Model 3 — One-step XGBoost surrogate and baselines

## Model details
- **Notebook:** `stage4-ml-shap-pipeline.ipynb`.
- **Type:** one `xgboost.XGBRegressor` per monitored room (seven models), predicting indoor air temperature one hour ahead. Hyperparameters fixed for all rooms: `max_depth = 6, n_estimators = 600, learning_rate = 0.03, subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1.0` (`ADR-022`).
- **Features:** current outdoor temperature and RH; outdoor-temperature lags of 1, 3, 6, 12 and 24 h; 24-h and 48-h rolling means; the measured previous-hour indoor temperature; cyclic hour-of-day and day-of-year terms.
- **Baselines (same data and split):** persistence (t−1); diurnal persistence (t−24); ARX (ridge regression on the same features); 1R1C grey-box model, T(t) = a·T(t−1) + (1 − a)·Tout(t) + gains, with time constant τ = −1/ln(a) hours; XGBoost without the indoor lag (weather-only).
- **Interpretation:** SHAP (TreeExplainer) on held-out samples.
- **Artefacts:** `2_output/models/stage4_xgb_<Room>.pkl` (joblib bundle with metadata) and `.json` booster; provenance in `2_output/tables/stage4_model_manifest.csv`.
- **References:** Chen & Guestrin (2016); Lundberg & Lee (2017).

## Intended use
- **Uses:** testing whether a tree-based surrogate adds predictive information beyond simple baselines; diagnosing where and why it fails; describing model-attributed feature contributions.
- **Out of scope:** autonomous forecasting (the measured previous-hour temperature is an input, so one-step metrics describe state-conditioned prediction, `ADR-024`); causal attribution from SHAP (`ADR-028`).

## Factors
Room (seven, each with its own chronological split, `ADR-021`), floor, the opposing-orientation bedroom pair, outdoor-temperature bin, and the upper tail of the indoor-temperature range.

## Metrics
- **One-step:** R², RMSE and MAE on the held-out years for every model; paired moving-block bootstrap intervals of RMSE differences.
- **Recursive:** RMSE by horizon over the 14-day heatwave window (against a held-seed persistence forecast) and over weekly rolling-origin 168-h forecasts through the held-out cooling seasons (bootstrapped over origins); split-conformal 90% intervals by horizon band with empirical coverage.
- **Extrapolation diagnostics:** RMSE and bias when the previous-hour indoor temperature is above the training 95th percentile or maximum; bias by outdoor-temperature bin; residuals in the hottest decile of the heatwave window.
- **Interpretation:** mean |SHAP| by room and floor; SHAP dependence on the 24-h outdoor mean; opposing-orientation contrast.

## Data
- **Training:** each room's earlier complete monitoring years (1 May–30 April) from Dataset A.
- **Evaluation:** the latest complete year and any later partial year; never used for hyperparameter selection.

## Quantitative analyses
`stage4_model_manifest.csv`, `stage4_model_performance.csv`, `stage4_rq2_onestep_vs_persistence.csv`, `stage4_benchmark_onestep.csv`, `stage4_rc_time_constants.csv`, `stage4_benchmark_upper_tail.csv`, `stage4_benchmark_recursive_rolling_origin.csv`, `stage4_conformal_coverage.csv`, `stage4_recursive_forecast_validation.csv`, `stage4_residual_analysis_heatwave.csv`, `stage4_onestep_bias_by_outdoor_bin.csv`, `hyperparameter_tuning_check_masterbed.csv`, `shap_feature_importance_by_room.csv`, `zone_comparison_ground_vs_upper_shap.csv`, `stage4_orientation_contrast_shap.csv`.

## Ethical considerations
As Model 1; the model has no demographic inputs.

## Caveats and recommendations
- Always report the surrogate against the baselines, not alone; strong temporal persistence makes high one-step accuracy easy to achieve.
- A tree ensemble cannot predict beyond its training targets, so it under-predicts when the test or future conditions are warmer than training; check the upper-tail and outdoor-bin tables before using it in hot conditions.
- SHAP values are associations among correlated features; the orientation contrast rests on two rooms.

[Back to top](#-oasis--model-card)

---

# 📈 Model 4 — Temperature-change (ΔT) XGBoost

## Model details
- **Notebooks:** `stage4-ml-shap-pipeline.ipynb` (Section 5e) and `stage5-exceedance-cdh-pipeline.ipynb` (year-long test).
- **Type:** the Model 3 configuration trained to predict the hourly change T(t) − T(t−1); the forecast is the previous temperature plus the predicted change, so it is not bounded by the training temperature range. Same features, training years and hyperparameters as Model 3 (`ADR-026`).

## Intended use
- **Uses:** a controlled alternative that isolates the effect of the prediction target on extrapolation; a model-form check on the DSY projections (Model 5).
- **Out of scope:** replacing the level model as the main surrogate; the level model remains the reference for the projections.

## Factors
Room, forecast horizon (1, 24, 72, 168, 336 h and a full year), outdoor-temperature bin and the heatwave window.

## Metrics
Protocol fixed before the comparison: comparators (persistence, level XGBoost, ARX, 1R1C; for multi-step also held-seed and diurnal persistence); RMSE, MAE, mean bias and daily-peak error; extreme-heat subsets (outdoor temperature above the training 90th percentile, outdoor bins, heatwave window); decision rule — ΔT outperforms a comparator only when the 95% paired bootstrap interval of the RMSE difference lies below zero. Year-long test (Stage 5): RMSE not higher than the level model and monthly mean bias within ±1.0 K.

## Data
As Model 3 (Stage 4); the chronological 80/20 split of Dataset A (Stage 5).

## Quantitative analyses
`stage4_dT_onestep.csv`, `stage4_dT_onestep_tests.csv`, `stage4_dT_hyperparameter_check.csv`, `stage4_dT_rolling_origin.csv`, `stage4_dT_rolling_origin_tests.csv`, `stage4_dT_heatwave.csv`, `stage4_onestep_bias_by_outdoor_bin_pooled.csv`, `stage5_dT_longrun_validation.csv`, `stage5_dT_longrun_monthly_bias.csv`.

## Ethical considerations
As Model 1.

## Caveats and recommendations
- Report its results with the horizon: an advantage at short horizons need not hold over long free-running forecasts.
- Accumulating predicted changes can drift; check the monthly bias table before using it for long runs.

[Back to top](#-oasis--model-card)

---

# 🔮 Model 5 — Recursive future-weather (DSY) projection

## Model details
- **Notebooks:** `stage4-ml-shap-pipeline.ipynb` (Section 8, evaluation models) and `stage5-exceedance-cdh-pipeline.ipynb` (production models).
- **Type:** the Model 3 surrogates run recursively, with each predicted temperature fed back as the lag input. Production models are refitted on the full record after an 80/20 recursive validation (`ADR-032`) and saved as `2_output/models/stage5_production_xgb_<Room>.pkl` and `.json`, with provenance in `stage5_model_manifest.csv`.
- **Projection set-up:** DSY2030 and DSY2080 weather (Dataset D) with 48-h circular padding for features; January mean indoor temperature as the seed, with 10th- and 90th-percentile seeds as a sensitivity test; feature-range audit against the training data.
- **Uncertainty and model form:** split-conformal 90% interval calibrated on odd ISO weeks of the test cooling season and checked on even weeks; recursive ARX and 1R1C benchmarks; exposure recomputed with weather-only XGBoost, ΔT-XGBoost, ARX and 1R1C; correction of each room's recursive bias by outdoor-temperature bin; thresholds of 25–28 °C.

## Intended use
- **Uses:** screening-level comparison of cooling-season exposure between climate horizons and rooms, for the household's own operation; identifying where further assessment is warranted.
- **Out of scope:** worst-case estimates; formal TM59 or Part O projections; absolute exposure values without their interval and model-form range.

## Factors
Room, climate horizon, model form, threshold, seed, and uniform warming of +1 to +3 °C (Stage 6).

## Metrics
- **Recursive validation:** R² and RMSE against held-seed persistence, ARX and 1R1C with paired block-bootstrap intervals; conformal coverage.
- **Projection:** hours and percentage of cooling-season hours (and occupied hours) above 26 °C with conformal bounds; TM59-style indicators (Stage 4); share of hours outside the training range; seed convergence.
- **Sensitivity:** exposure by model form; bias-corrected exposure; exposure at 25–28 °C.

## Data
Training: Dataset A (full record for production models). Projection: Dataset D.

## Quantitative analyses
`stage5_data_quality_audit.csv`, `stage5_model_manifest.csv`, `stage5_temporal_validation.csv`, `stage5_production_model_fit.csv`, `stage5_recursive_benchmarks.csv`, `stage5_conformal_interval.csv`, `stage5_future_feature_domain_audit.csv`, `stage5_seed_sensitivity.csv`, `stage5_exceedance_hours_by_scenario.csv`, `stage5_exceedance_mean_per_room.csv`, `stage5_exposure_prediction_interval.csv`, `stage5_exposure_prediction_interval_mean.csv`, `stage5_exposure_by_model.csv`, `stage5_recursive_bias_by_outdoor_bin.csv`, `stage5_bias_corrected_exposure.csv`, `stage5_threshold_sensitivity.csv`, `stage5_threshold_sensitivity_mean.csv`; Stage 4: `dsy_input_quality_audit.csv`, `dsy_outdoor_weather_discrepancy.csv`, `dsy_ml_surrogate_projection_summary.csv`, `dsy_feature_domain_extrapolation_audit.csv`, `dsy_physics_vs_ml_matched_rooms.csv`, `stage4_physics_vs_ml_tm59.csv`.

## Ethical considerations
As Model 1.

## Caveats and recommendations
- The surrogates under-predict the hottest hours; uncorrected projections understate exposure, and the bias correction is constant within each outdoor-temperature bin.
- The conformal interval reflects errors on measured data, not errors under conditions outside the record.
- Room rankings can change with model form; quote room-level results with the model-form comparison.
- DSY2030 and DSY2080 are single realisations, not a description of climate uncertainty.

[Back to top](#-oasis--model-card)

---

# 🧩 Model 6 — Scenario layer: Passive, Hybrid and Active

## Model details
- **Notebook:** `stage5-exceedance-cdh-pipeline.ipynb` (Steps 5.1–5.3); screening summaries in `stage6-ml-ai-advanced-synthesis.ipynb` (Section 6.2).
- **Type:** deterministic post-processing of the Model 5 free-running projection; not a simulation.
  - **Passive** and **Hybrid** — cross-model intervention sensitivity experiments: for each room, horizon and (month, hour-of-day) pair from May to September, the EnergyPlus as-built temperature minus the Case 2A (Passive) or Case 3B (Hybrid) temperature is subtracted from the projection; the Hall and Landing take the mean profile of simulated rooms on the same floor (`ADR-036`).
  - **Active** — indoor temperature ceilings of 18, 20, 22 and 24 °C, with cooling degree-hours (CDH, °C·h) above each ceiling (`ADR-037`).

## Intended use
- **Uses:** showing how sensitive the screening metric is to an assumed intervention response, and how the cooling-demand proxy changes between horizons and set points.
- **Out of scope:** intervention or retrofit performance in the household; energy, carbon or cost estimates; HVAC sizing.

## Factors
Scenario, room, climate horizon, set point and offset definition.

## Metrics
Hours above 26 °C and reductions relative to free-running (reported both as the reduction in the dwelling mean and as the mean of per-room reductions — check which definition a table uses); hours above each ceiling and CDH.

## Data
Model 5 projections; Dataset C (offsets).

## Quantitative analyses
`stage5_intervention_assumptions.csv`, `stage5_scenario_reduction_dwelling_mean.csv`, `stage5_exposure_interval_by_scenario.csv`, `stage5_cdh_by_room.csv`, `stage5_active_setpoint_sensitivity.csv`, `stage5_cdh_climate_change.csv`, `Stage6_Intervention_Screening.csv`, `Stage6_Active_SetPoint_Screening.csv`.

## Ethical considerations
The main risk is false precision. CDH is never converted to energy, and Passive/Hybrid results are labelled as sensitivity experiments.

## Caveats and recommendations
- The transfer assumes the simulated reductions are additive to the surrogate projection and carry over from TM59 operation to the household's operation; neither has been tested.
- An Active ceiling at or below 26 °C removes all hours above 26 °C by construction.
- Results inherit every caveat of Model 5.

[Back to top](#-oasis--model-card)

---

# 🧭 Model 7 — Room clustering and screening tiers

## Model details
- **Notebook:** `stage6-ml-ai-advanced-synthesis.ipynb` (Sections 6.0–6.3).
- **Type:**
  - **Clustering:** K-means with a Ward hierarchical cross-check on seven rooms, using projected hours above 26 °C (DSY2030, DSY2080), CDH above 24 °C (DSY2080), peak indoor temperature (DSY2080) and thermal gain from the warming stress test (`ADR-042`, `ADR-043`).
  - **Room Vulnerability Index:** unweighted mean of the min–max-normalised clustering features, used only to order cluster labels (`ADR-044`).
  - **Screening tiers:** Low (≤ 1%), Elevated (> 1–3%), High (> 3–10%) and Severe (> 10%) of cooling-season hours above 26 °C, shown with the hot-hour confidence flag and model-instance spread (`ADR-045`).
  - **Reliability diagnostics:** confidence flag from the mean residual over the hottest decile of outdoor-temperature hours in the heatwave window (high |bias| < 0.5 K, medium 0.5–1.5 K, low ≥ 1.5 K); spread between Stage 4 and Stage 5 models; uniform-warming stress test.
  - **Hazard characterisation:** peak outdoor temperature, hours above outdoor thresholds (including 28 °C, `EXTREME_C`) and the duration of persistent warm periods in each DSY file.

## Intended use
- **Uses:** exploratory within-dwelling synthesis; a room-by-scenario matrix indicating where further engineering assessment is warranted.
- **Out of scope:** a transferable archetype classifier; a compliance or health-risk classification.

## Factors
Room, k (2 and 3 compared), scenario, climate horizon, surrogate form and conformal bound.

## Metrics
Silhouette, ARI against Ward, smallest cluster size and leave-one-room-out stability, with bootstrap intervals; tier per room, scenario and horizon; number of room–scenario–horizon cases with an unchanged tier, and rank correlation with the main surrogate, for each alternative surrogate and at each conformal bound.

## Data
Stage 4 and 5 outputs and model bundles; Dataset D for the stress test and hazard characterisation. No train/test split.

## Quantitative analyses
`Stage6_Model_Set_Overview.csv`, `Stage6_Surrogate_Reliability.csv`, `Stage6_Model_Instance_Spread.csv`, `Stage6_Warming_Stress_Test.csv`, `Stage6_Warming_Sensitivity.csv`, `Stage6_Room_Thermal_Archetypes.csv`, `Stage6_Cluster_Comparison_k2_k3.csv`, `Stage6_Hazard_Characterisation.csv`, `Stage6_Room_Risk_Matrix.csv`, `Stage6_Estate_Resilience_Synthesis.csv`, `Stage6_Tier_Sensitivity.csv`.

## Ethical considerations
Tiers are never shown without their reliability context, so that a high exposure in a low-confidence room is not read as an established result.

## Caveats and recommendations
- With seven rooms, use the partition that is stable under every check as the reference and treat others as descriptive.
- Clusters and tiers come from the main surrogate's projections; check `Stage6_Tier_Sensitivity.csv` before reporting a room ranking, because rankings can change with model form.
- The vulnerability index and tiers are transparent screening constructs, not a validated risk score.

[Back to top](#-oasis--model-card)
