# Public input data

The study uses the HI-Small and LI-Small subsets of IBM's synthetic AML data,
attributed in the manuscript to Altman et al. (2023). The original distribution
used by the research preprocessing has the Kaggle identifier:

`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`

Obtain the transaction CSVs and corresponding laundering-pattern annotations
from that distribution under its terms. Raw transactions, account identities,
labels, feature matrices and annotations are not republished here.

`dataset_statistics/` contains the aggregate counts retained by the study.
The initial preprocessing maps bank/account pairs to compound account IDs,
normalizes sender/receiver and amount/currency/time columns, and marks an
account positive if it participates in at least one labelled laundering
transaction. See `Altman_experiments/code/ibm_pipeline.py` for the implementation.

For a full rerun, the research scripts expect normalized parquet files under
`Altman_experiments/data/{hi_small,li_small}/` and derived feature/graph caches
under their corresponding result stages. These are generated inputs, not part
of the compact GitHub package. Exact feature names, candidate lists and split
seed formulas are retained in the included protocol/registry JSON files.
