
# MMA3001 – Building Occupancy Prediction Using Machine Learning

## Project Overview

This project investigates whether machine learning can predict short-term occupancy patterns in a university building using historical occupancy sensor data.

The objective is to predict the **occupancy fraction of the next 15-minute interval** using recent occupancy measurements and time-related features.

These predictions could potentially support occupancy aware heating, ventilation and air-conditioning (HVAC) scheduling, although this project does not directly model HVAC energy consumption or control HVAC equipment.


## Engineering Problem

Building HVAC systems may operate according to fixed schedules that do not always reflect actual building usage.

Predicting future occupancy could provide useful information for adjusting ventilation and conditioning schedules.

However, real-world occupancy sensor data presents several challenges, including:

- Irregular reporting intervals.
- Extended periods without sensor readings.
- Uncertainty about actual room occupancy.
- Changes in occupancy patterns across different times and days.

This project develops and evaluates a numerical and machine learning approach to address these challenges.

## Dataset and Preprocessing

The project uses timestamped occupancy sensor readings from a university building.

The primary analysis covers **22 July to 18 October 2024**, corresponding to a selected teaching-period dataset.

Sensor states were converted into binary occupancy indicators:

- `CurrentlyOccupied` → 1
- `RecentlyOccupied` → 0
- `NotOccupied` → 0

The data was then processed using the following steps:

1. Sort sensor measurements chronologically.
2. Resample irregular readings into one-minute intervals.
3. Carry forward the most recent sensor state for a maximum of five missing minutes.
4. Calculate the mean occupancy indicator within each 15-minute interval.
5. Retain intervals containing at least 12 valid or filled one-minute values.
6. Generate input features and the next-interval prediction target.

The resulting occupancy fraction is an **estimate derived from sensor states**, not a verified physical headcount or independently measured room occupancy rate.

## Model Inputs and Output

The models use the following input features:

| Input | Description |
|---|---|
| Current occupancy | Estimated occupancy fraction in the current 15-minute interval |
| Previous occupancy | Estimated occupancy fraction in the preceding interval |
| Hour | Hour of the next prediction interval |
| Day of week | Day associated with the next prediction interval |
| Weekend | Whether the next interval occurs on a weekend |

**Output:** Predicted occupancy fraction for the following 15-minute interval, bounded between 0 and 1.

## Prediction Methods

Three approaches were evaluated:

1. **Persistence baseline:** Assumes the next occupancy fraction will equal the current fraction.
2. **Linear Regression:** Models the relationship between input features and future occupancy.
3. **Decision Tree Regression:** Captures nonlinear relationships between features and occupancy.

The dataset was divided chronologically into training, validation and test periods, with exclusion gaps around the split boundaries to reduce temporal leakage.

Decision Tree depth was investigated using validation data to assess model complexity and overfitting.

## Model Performance

The final test results were:

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Persistence baseline | 0.1262 | 0.1959 |
| Linear Regression | 0.1315 | 0.1838 |
| Decision Tree Regression | 0.1304 | 0.1830 |

Linear Regression was selected as the preferred model based on its validation RMSE and interpretability.

Although the Decision Tree achieved a slightly lower RMSE on the final test dataset, the preferred model was not changed after examining the test results.

These results indicate that the machine learning models reduce larger prediction errors compared with the persistence baseline, while the baseline remains competitive on mean absolute error.

## Verification, Sensitivity and Computational Performance

Verification included checking:

- Occupancy fractions and predictions remain within valid bounds.
- The prediction target corresponds to the following 15-minute interval.
- Chronological ordering is maintained.
- Training, validation and test data are separated appropriately.
- Repeated preprocessing and model fitting produce consistent results.

A sensitivity analysis compared carry-forward limits of **2, 5 and 10 minutes** to investigate how assumptions about missing sensor readings affect the resulting dataset and model performance.

Computational profiling was also conducted to compare training and prediction times.

## Automated Testing

Automated tests were implemented using `pytest`.

All **8 tests passed**, including checks for preprocessing, invalid input parameters, model prediction bounds, reproducibility and evaluation calculations.

- Test functions: [`tests/test_project.py`](tests/test_project.py)
- HTML test report: [`results/test_report.html`](results/test_report.html)

## Repository Structure

```text
MMA3001-Occupancy-Prediction/
├── notebooks/
│   └── occupancy_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
├── tests/
│   └── test_project.py
├── results/
│   └── test_report.html
├── docs/
│   └── html/
│       ├── index.html
│       ├── search.js
│       └── src/
├── README.md
└── .gitignore
```

## Running the Project

The project was developed using Python in Google Colab.

To inspect or reproduce the analysis:

1. Clone or download this repository.
2. Open `notebooks/occupancy_prediction.ipynb` in Google Colab.
3. Install the required Python dependencies.
4. Obtain the original occupancy sensor dataset through an authorised source.
5. Update the notebook's dataset path to point to the available local or Google Drive file.
6. Run the notebook cells in order.

The raw sensor dataset is not included in this repository. Reproducing the complete analysis requires access to the original data.

## Source Code Documentation

HTML API documentation was generated using `pdoc` from the Python source modules.

Documentation is stored in [`docs/html/index.html`](docs/html/index.html).

The documentation describes the preprocessing, modelling and evaluation functions, including their inputs and outputs.

## Limitations and Future Work

Important limitations include:

- The sensor measurements provide an approximate indication of occupancy rather than verified headcount.
- Irregular reporting and missing measurements can affect reconstructed occupancy fractions.
- The analysis uses selected data from one monitored location and period.
- Prediction accuracy may differ for other buildings, semesters or occupancy patterns.
- The models have not been evaluated using an actual HVAC control system.

Potential future improvements include incorporating additional sensors, evaluating longer periods, investigating more advanced forecasting models and testing predictions against independent occupancy measurements.

## Licence and Data Availability

The project source code and supporting documentation are provided for academic assessment. 

The original sensor dataset is not redistributed with the repository. Access to the data is subject to the relevant data owner's permissions.
