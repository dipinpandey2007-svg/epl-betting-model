"""Market-implied probability benchmark (protocol market_benchmark_v1).

- odds: which raw columns hold which snapshot, and reading them without any result column;
- devig: implied probabilities and the margin-removal methods (Shin, proportional, power);
- coverage: validity rules and missing-data accounting;
- benchmark: forecast frames of the market arms, in the format of eplmodel.evaluation.forecasts.
"""
