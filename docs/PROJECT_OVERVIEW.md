# EPL Betting Model — Project Overview

## Goal

Build a serious, research-grade football betting probability model for the English Premier League.

The long-term objective is NOT simply to predict match winners. The objective is to investigate whether genuine betting-market inefficiencies can be identified.

The project will progressively develop:

1. Historical data pipeline
2. Elo baseline
3. Poisson / Dixon-Coles goal model
4. xG and team-performance features
5. Squad, injury and lineup information
6. Tactical and contextual features
7. Market odds benchmark
8. Machine-learning models
9. Probability calibration
10. Walk-forward / backtesting framework
11. Closing-line-value tracking
12. ROI and proper scoring metrics
13. Structured human football knowledge

## Research principles

- Avoid look-ahead bias and data leakage at all costs.
- Every prediction must use only information available before kickoff.
- Use chronological, out-of-sample and walk-forward evaluation where appropriate.
- Compare models against market-implied probabilities once market data is incorporated.
- Focus on probability quality, calibration, log loss, Brier score, closing-line value and long-run performance.
- Do not focus only on classification accuracy.
- Do not recommend real-money betting during model development.
- Be skeptical of apparent improvements.
- Distinguish training fit from genuine predictive performance.
- I am learning Python and statistics through this project, so explain important methodological decisions rather than treating the project as a black box.

## Development philosophy

Build progressively.

Do not jump directly to XGBoost, neural networks or complex features before simpler baselines are correctly implemented and evaluated.

Do not move to the next major modeling stage until the current stage has been properly validated.

When a methodological choice is debatable, explain the trade-off and flag weaknesses rather than silently choosing an approach.
