# Final Project

## Goal
This repos aims to test empirical performance of Theoroem 1 of this paper "Eliciting User Preferences for Personalized Multi-Objective Decision Making through Comparative Feedback"
https://arxiv.org/abs/2302.03805

## Parameters

- $\theta$ : tolerance threshold to control accuracy of ValueIteration and PolicyEvaluation
- $\gamma$ : discounting factor
- noise : stochasticity in state transition
- $\epsilon$: indistinguishability
- $K$: number of objectives
- rep: number of repetition

## Usage
- Python RunMDP.py (it takes around 1.5 h)
- Python GenResult.py

## Files
- GridWorldMDP.py: Multi-Objective GridWorld Environment, ValueIteration & Policy Evaluation
- RunMDP.py: Estimate user's preference
- GenResult.py: Generate tables and figures

