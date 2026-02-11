
# Technical Documentation: Predictive Maintenance System

## Project Overview

This repository contains a comprehensive machine learning solution for industrial predictive maintenance. The system utilizes the `predictive_maintenance.csv` dataset to analyze sensor data and predict machine failures before they occur. It focuses on classifying five distinct failure types using the XGBoost algorithm, providing a robust tool for Industry 4.0 applications.

## Technical Objectives

Implement a multi-class classification model for high-accuracy failure diagnostics.
Address class imbalance in industrial data using the SMOTE technique.
Deliver an interactive web-based dashboard for real-time monitoring and analytics.
Provide deep-dive physical insights through automated data visualization.

## Dataset Description

The system processes the `predictive_maintenance.csv` file, which includes 10,000 data points with the following key attributes:

Rotational Speed (RPM): Measures the mechanical velocity of the machine.
Torque (Nm): Indicates the rotational force applied.
Air and Process Temperature (K): Tracks the thermal conditions of the operation.
Tool Wear (min): Records the cumulative usage time of the machining tool.

## Methodology

### 1. Data Preprocessing

Feature Engineering: Encoding product quality types (Low, Medium, High).
Imbalance Handling: Implementation of SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model accurately identifies rare failure events.
Normalization: Scaling features to ensure uniform influence on the model's loss function.

### 2. Machine Learning Pipeline

The core engine utilizes XGBoost (Extreme Gradient Boosting),
    selected for its efficiency with tabular data and its ability to capture complex non-linear relationships between mechanical sensors.

## System Architecture

Backend: Flask (Python) serving as the inference engine.
Frontend: HTML/Bootstrap dashboard for displaying predictions and analytics.
Storage: Local file system for saving dynamically generated charts in the `static/` directory.

## Analytics and Visualizations

The system automatically generates five strategic charts to assist engineers in root-cause analysis:

1.Speed vs. Torque Analysis: Identifies mechanical operating limits.
2.Thermal Regression Plot: Correlates process temperature with failure probability.
3.Tool Wear Distribution: Analyzes degradation across different product categories.
4.Thermal Density (KDE): Maps the overlap between air and process temperatures.
5.Correlation Heatmap: Provides a statistical overview of sensor interdependencies.

## Installation and Deployment

1.Environment Setup:
Ensure Python 3.8+ is installed.
2. Install Dependencies:
```bash
pip install -r requirements.txt

```


3.Execution:
```bash
python app.py

```


The application will be accessible at `http://127.0.0.1:5000`.

## Project Structure

`app.py`: Main application script for the Flask server and model inference.
`predictive_maintenance.csv`: Primary dataset.
`static/`: Directory containing generated PNG charts and CSS.
`templates/`: HTML templates for the web interface.
`requirements.txt`: List of necessary Python libraries.

## Credits

Instructor: Eng. Ayman
Institution: Al-Nasser UN
Development Year: 2026

