# Tests directory

This folder contains all files related to testing the hospital-stay-discharge-prediction pipeline.

## Contents

- **Unit and integration tests:**  
  All test scripts (e.g., `test_feature_processing.py`) for validating the pipeline’s functions and modules.
- **Sample data:**  
  All sample data and expected outputs for local testing are inside the [`subsets`](./subsets) folder.  
  Use these smaller subsets to quickly try out the pipeline without needing the full dataset.
- **Expected outputs:**  
  Example outputs for sample data, also located in the `subsets` folder, useful for verifying correctness.
- **Sample data creation script:**  
  [`create_raw_data_subsets.py`](./create_raw_data_subsets.py) generates small sample CSVs from the full raw datasets for use in local testing.  
  Run this script to create or refresh the sample files in the `subsets` folder.

## Running Tests

- **Test framework:**  
  All tests use [pytest](https://docs.pytest.org/).
- **How to run:**  
  Always run pytest from the project root directory:
  ```bash
  pytest
  ```
  This ensures all imports and paths work correctly.

## Creating sample data

To generate or update the sample lab and clinical data used in tests, run:

```bash
python create_raw_data_subsets.py
```

This will create `sample_lab.csv` and `sample_main.csv` in the `subsets` folder, using the first 100,000 rows of the lab data and 20,000 rows of the clinical data.

## Adding tests

- Place new test scripts in this folder.
- Name test files and functions with the `test_` prefix so pytest can discover them.
- Add any new sample data or expected output files to the `subsets` folder.

## Notes

- Sample data in the `subsets` folder is for development and debugging only.  