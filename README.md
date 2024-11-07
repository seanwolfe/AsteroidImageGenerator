# Asteroid Synthetic Data Generation

This repository contains a set of scripts to generate synthetic data for asteroid observations. The scripts use real astronomical data, apply specific statistical distributions, and perform image manipulation to create synthetic datasets for machine learning purposes.

## Overview

### Main Modules

1. **`BackgroundGen_par_stack.py`**: Generates stellar backgrounds using real astronomical images. It selects random images, preprocesses them, and crops specific regions for later use in synthetic data creation.
2. **`DistribuGen.py`**: Queries the JPL Horizons database to retrieve parameters of known Near-Earth Asteroids (NEAs) and generates statistical distributions based on these parameters.
3. **`SnrGen.py`**: Calculates the expected Signal-to-Noise Ratio (SNR) for an asteroid, considering telescope specifications and observational parameters.
4. **`ImageGen_par_stack.py`**: Combines all elements to generate synthetic image stacks with specified SNR values. Runs in parallel using MPI to speed up data generation.
