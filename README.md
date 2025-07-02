# PINN-Based Aerodynamic Flow Prediction and Clustering 🚗

This repository contains code and experiments for predicting 3D aerodynamic flow around car geometries using Physics-Informed Neural Networks (PINNs) and for analyzing flow features via autoencoder-based clustering.

## Features

- ✅ **Physics-Informed Neural Networks (PINNs)**  
  Predict 3D velocity fields around car shapes without full CFD simulations.

- ✅ **Streamline Visualization**  
  Visualize predicted flow fields and identify regions of interest.

- ✅ **Autoencoder for Feature Compression**  
  Reduce high-dimensional CFD data to meaningful latent representations.

- ✅ **KMeans Clustering in Latent Space**  
  Detect and classify distinct aerodynamic zones like wakes, stagnation points, and freestream regions.

- ✅ **3D Visualization of Clusters**  
  Overlay cluster labels on mesh geometry for intuitive interpretation.

## Use Cases

- Aerodynamic analysis and design exploration  
- Data-driven identification of critical flow regions  
- Reducing reliance on computationally expensive CFD runs

## Technologies

- PyTorch  
- PyVista  
- scikit-learn  
- NumPy

## Authors

- Mariam Abidi — PINN & Autoencoder & clustering implementation 
- Suhas Vittal — PINN implementation, streamline visualizations 
- Nishith Hingoo — Dataset sourcing, preprocessing pipelines

---

> This project demonstrates the feasibility of physics-guided machine learning for aerodynamic analysis and provides a framework for faster, simulation-free flow predictions and feature detection.

