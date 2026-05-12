---
title: "F1 Lap Time Forecasting (MLP and LSTM)"
author: "Jarod DeFilippo"
date: "May 2026"
---

## Abstract
This project predicts F1 lap times from a driver's most recent ten laps for their next five laps in a race. Two full seasons of race data (2024–2025) along with the most up-to-date races in 2026 were used. The data was collected from the FastF1 Python library. Each lap was represented by a normalized lap-time delta against the median race pace, tyre compound, driver position, weather, and pit-stop indicators. A Multilayer Perceptron and a Long Short-Term Memory network were trained on 10-lap input windows and compared via RMSE using a 75% training / 25% test split. Both models converged to a similar test RMSE around 7.5 seconds, but both suffered from overfitting. This overfitting was driven by the unpredictability of pit-stop laps within each window.

## Introduction
Formula 1 races last between one and a half and two hours, with the number of laps ranging from 44 to 78. There are currently eleven teams on the grid (2026 season), but in the 2024–2025 seasons there were only ten — Audi replaced Sauber for 2026 and Cadillac is a completely new team. Lap times can vary greatly due to tyre wear, fuel load, traffic, weather, and pit-stops. Pit-stops introduce the most variation, as teams strategically time them to maximize their drivers' position.

This project predicts the next five laps for a driver based on their previous ten laps as a sequence-to-sequence regression. Each lap is represented by 29 features: lap time relative to race median (LapTimeDelta_s), tyre compound (one-hot), tyre life, stint number, race position, pit-stop indicators, weather (air/track temperature, humidity, wind, rainfall), team, and year. The target is the next five lap-time deltas (relative to the race median), a 5-dimensional regression output.

## Methods
### Data Collection
All race data was downloaded via the FastF1 Python package. Only race sessions were collected (no free practice, qualifying, or sprint races) for the past three seasons (2024, 2025, 2026). The other sessions were excluded because their lap times are not under race conditions (different fuel loads, different driver objectives).

For each lap, the following were extracted: driver, team, lap number, stint number, raw lap time, sector times, tyre compound, tyre age, grid position, track-status flag, pit-in/pit-out times, and per-lap weather (air temperature, track temperature, humidity, atmospheric pressure, rainfall, wind speed, and wind direction). Telemetry data was also available but not used, as it would drastically increase the size of the dataset without adding clear predictive power at lap-level resolution. The raw dataset contained 57,366 lap rows across 52 race sessions.

### Data Cleaning
Three teams spanning the 2024–2026 seasons introduced complications:

- Kick Sauber (2024–2025) was rebranded as Audi for 2026.
- Cadillac is a brand-new team in 2026 with no historical data.
- RB was renamed to Racing Bulls in 2025, but it is still the same team.

To handle this, only the teams that competed across all three seasons were kept: Red Bull Racing, Ferrari, McLaren, Mercedes, Williams, Haas, Aston Martin, Alpine, and Racing Bulls. Audi/Kick Sauber and Cadillac rows were dropped, while RB was renamed to Racing Bulls. This removed 5,904 lap rows, leaving 51,462.

### Target Variable
Raw lap time was not used as the target, as track-specific baselines (Monaco's ~75-second laps vs. Spa's ~105-second laps) plus regulation differences (2026 cars are slower) would force the model to memorize per-track and per-year offsets. Instead, lap time relative to the race's median pace was defined:

LapTimeDelta_s = LapTime_s − RaceMedian_s

The race median was computed from a clean subset of laps (not pit-in/out, green flag, non-null lap time). The intent was that the model predicts pace patterns under normal racing conditions. In the dataset, pit laps appear as large positive deltas (+20 to +30 seconds) and fastest laps as small negative deltas.

### Outlier Cleaning
After computing the delta target, two further cleaning steps were applied:

- 562 rows had no valid delta (their race had no clean baseline laps) and were dropped.
- 70 rows had |delta| > 60 seconds, indicating data corruption (red-flag periods where the lap timer kept running) and were dropped.

Pit-stop laps were not dropped — they are real racing condition laps the model should learn to predict. The final cleaned dataset contained 50,830 lap rows.

### Feature Engineering and Normalization
The 14 raw columns were transformed into 29 numeric features:

- **Numerics (8):** LapTimeDelta_s, TyreLife, Position, Stint, AirTemp, TrackTemp, Humidity, WindSpeed. The seven non-target numerics were standardized to $\mu=0, \sigma=1$ using train-set statistics.
- **Booleans (4):** IsPitInLap, IsPitOutLap, FreshTyre, Rainfall, cast to 0.0 / 1.0.
- **One-hot categoricals (17):** tyre compound (5: SOFT/MEDIUM/HARD/INTERMEDIATE/WET), team (9), year (3: 2024/2025/2026).

### Train/Test Split
The 52 races were randomly split 75/25 by race (39 training races, 13 testing races). The split was performed at the race level to prevent the same (year, race, driver) sequences from appearing in both train and test, which would inflate test performance.

### Windowed Dataset
A sliding-window dataset, LapWindowDataset, was constructed for each (year, round, driver) sequence. Each window had a lookback of ten laps and a horizon of five laps. Windows never spanned across races or drivers, producing 28,254 training and 10,076 test windows. Windows were batched at size 128, with shuffling on the training loader.

### Model Architectures
**MLP (23,109 parameters).** Flattens the (10, 29) window to a 290-vector, then passes through two hidden layers of 64 neurons each with ReLU activations and dropout (p=0.2), ending in a Linear layer that outputs the next 5 laps. Weights used He/Kaiming normal initialization; biases initialized to zero.

**LSTM (82,053 parameters).** Single-layer LSTM with hidden dimension 128, processing the (10, 29) input window. The final hidden state is fed into a dropout layer (p=0.2) and then into a Linear layer for the 5-lap output. Weights initialized with Xavier/Glorot uniform; biases initialized to zero.

### Training
Both models were trained with identical hyperparameters:

- **Loss:** mean squared error, reported as RMSE for interpretability.
- **Optimizer:** Adam, learning rate = 0.001.
- **Weight decay:** L2 regularization with $\lambda = 0.001$.
- **LR schedule:** StepLR with step size = 10, $\gamma = 0.5$, halving LR every 10 epochs.
- **Batch size:** 128.
- **Epochs:** 30.

## Results and Discussion

### Exploratory Data Analysis

#### Race pace by tyre compound (Verstappen, Bahrain 2024)
![Verstappen race pace](figures/verstappen_pace_scatter.png)

- Three distinct stints are visible
- Lap times increase as tyres degrade

#### Team pace distribution (Bahrain 2024)
![Team pace](figures/team_pace_boxplot.png)

- Red Bull Racing has the lowest median lap time
- Wide interquartile range (different drivers within the same team)
- Each team has a distinct pace

#### Tyre strategy
![Tyre strategy](figures/tyre_strategy.png)

- Different pit stops for each driver
- Hülkenberg had four stops due to a first-lap incident
- Pit timing is sometimes unplanned

#### Pace distribution by season
![Pace distribution by year](figures/pace_distribution_by_year.png)

- The KDE plot shows that all three seasons share a similar distribution of lap-time deltas
- This is encouraging — the model does not need to learn the regulation differences between seasons

### Model Comparison
![Loss curves](figures/loss_curves.png)

- The MLP model performed better due to less overfitting, but still present
- Both models were already overfitting from the second epoch

#### Tolerance accuracy
![Tolerance accuracy](figures/tolerance_accuracy.png)

- Both models have about half of all per-lap predictions within a one-second tolerance
- As the tolerance increases, the fraction of predictions within +/- delta rises sharply
- The asymptote near 85–90% reflects the unpredictable pit-stop laps in the prediction horizon

#### Predicted vs actual
![Predicted vs actual](figures/pred_vs_actual.png)

- Both models have clusters near predicted delta = 0
- The models seem to be predicting the near-median pace but not accounting for pit stops

## Conclusions
Training both models on this Formula 1 dataset resulted in similar performance: both converged to a test RMSE around 7.5 seconds. The bottleneck was that test windows often contain pit-stop laps with deltas of +20 to +30 seconds, but past lap-time data does not encapsulate when a driver is about to pit. Pit timing is a strategic team decision, and ten laps of lookback simply does not contain enough information for the model to anticipate it. The tolerance accuracy curves and predicted-vs-actual plots showed this clearly.

To address this while keeping a similar architecture, two extensions are worth pursuing:
1. Add a pit-stop classifier as a parallel output head.
2. Add features the past lap times cannot encode — available tyre compounds, expected pit-stop count for the track, and pit-window indicators.

The main takeaway is that different architectural designs cannot mitigate a lack of strong predictive signal in the data. Considerable time was spent tuning the models, only to see worsened performance or minimal gains attributable to noise.

## References
- FastF1 — Python package used for data download
  - Years used: 2024–2026 (races only; no FP, qualifying, or sprints)
  - https://docs.fastf1.dev/index.html
- FastF1 — styling conventions for plots
  - https://docs.fastf1.dev/gen_modules/examples_gallery/index.html
