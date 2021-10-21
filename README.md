# Data Driven Modelling of Centrifugal Compressor Maps for Control and  Optimization Applications  

This repository hosts the source code of the paper: "Data Driven Modelling of Centrifugal Compressor Maps for Control and  Optimization Applications".

## How to use?

As explained in the paper, the proposed model approximates compressor efficiency and pressure ratio using Gaussian Process with Bayesian Optimisation, and Polynomial Regression. GP2D_scaled.py produces optimised kernels for each defined kernel type in sklearn with their hyperparameters and error metrics comparing to Polynomial regression.  It predicts efficiency or pressure ratio over validation data with the best-performed model and saves the results.  Different sample sizes and sampling methods may be assigned to analyse the performance of distinct models.

Comparison of different polynomial degrees and GPR performance is also possible using handcrafted_models.py.  Line plots with uncertainty quantification can be obtained using this function by setting the "plot" parameter to True.

Contour plots for representing approximated efficiency map, scatter plot to see speed lines produced by pressure ratio approximation of GP and their respective shape to true ones, and surface plots of both GP and Polynomial regression can be obtained by plot.py

## Available datasets

Currently, following datasets are ready to use in pressure ratio and efficiency regression.

| Compressor name | Data source name |
| --------------- | ---------------- |
| EFR 91S74       | etas_full        |
| Garrett 3076R   | garrett_full1    |
| Garrett 1544    | garrett_full2    |
| Garrett TO4B    | garrett_to4b     |

## Example estimated compressor efficiency maps

**EFR 91S74:**

| GPR        | Polynomial regression |
| -----------| --------------------- |
|<img src="https://user-images.githubusercontent.com/24464017/132126603-2960bf7b-8051-4a9d-a476-6cab537d07cf.png" alt="Contour_plot_of_GPR_etas_full-1" align="center" width="87%"/>|<img src="https://user-images.githubusercontent.com/24464017/132126668-8772917b-135b-4714-b6e7-979531d6cad9.png" alt="Contour_plot_of_GPR_etas_full-1" align="center" width="80%"/>|

**Garrett 1544:**

| GPR        | Polynomial regression |
| -----------| --------------------- |
|<img src="https://user-images.githubusercontent.com/24464017/132128275-f80bcaa5-f988-413a-898e-4632782c288f.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="87%"/>|<img src="https://user-images.githubusercontent.com/24464017/132128297-c0ddb281-a3a4-42e2-9a2d-3b1b429cd21f.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="80%"/>|

**Garrett 3076R:**

| GPR        | Polynomial regression |
| -----------| --------------------- |
|<img src="https://user-images.githubusercontent.com/24464017/132128389-fb3c517c-bbe3-4b4e-a194-655e920e07a2.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="87%"/>|<img src="https://user-images.githubusercontent.com/24464017/132128479-db2e6016-024f-4f8c-a433-1be50526f20f.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="80%"/>|

**Garrett TO4B:**

| GPR        | Polynomial regression |
| -----------| --------------------- |
|<img src="https://user-images.githubusercontent.com/24464017/132128713-fddd62ec-7ca4-42e2-b103-235631232ca0.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="87%"/>|<img src="https://user-images.githubusercontent.com/24464017/132128734-5136a62d-4891-49ee-8490-9aa3082214f8.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="80%"/>|

## Example estimated speed lines

**Garrett 1544:**

<img src="https://user-images.githubusercontent.com/24464017/132135217-8740372e-dedf-48e4-bbaa-70cd879f3624.png" align="center" width="60%"/>

**Garrett 3076R:**

<img src="https://user-images.githubusercontent.com/24464017/132135220-98a3c5cb-0e88-4cf7-b402-25c9691fb7f5.png" alt="Contour_plot_of_GPR_garrett_full-1" align="center" width="60%"/>

**Garrett TO4B:**

<img src="https://user-images.githubusercontent.com/24464017/132135196-e874ed0b-d726-4688-9370-947412846a9a.png" align="center" width="60%"/>

