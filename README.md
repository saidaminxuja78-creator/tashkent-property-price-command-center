# Tashkent Property Price Command Center

A Pearson BTEC Level 6 applied AI capstone prototype for estimating apartment asking prices from OLX Uzbekistan real-estate listings.

## Project purpose
The application is designed as an auditable real-estate decision-support prototype. It cleans scraped OLX listing data, trains regression models, compares them with a dummy baseline, explains price drivers, supports scenario simulation and exports evidence for academic assessment.

## Main features
- Data audit and cleaning workflow
- Price distribution and district-level market insights
- Regression model comparison against a dummy baseline
- Metrics: MAE, RMSE, R², MAPE, within-10% and within-20% accuracy bands
- Price prediction form for a single apartment profile
- Scenario simulator for valuation sensitivity
- Permutation-based explainability
- Evidence export package

## Dataset
The included dataset is based on OLX Uzbekistan real-estate listings. It is an asking-price dataset, not an official transaction registry. Therefore, predictions should be treated as market-support estimates rather than formal property valuations.

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Recommended Streamlit Cloud settings
- Repository: your GitHub repository
- Branch: main
- Main file path: app.py

## Responsible-use warning
This app is an academic prototype. It should not be used as the sole basis for buying, selling, lending or legal valuation decisions.
