SINDy-Based Modal Reduction (Sparse Identification of Nonlinear Dynamics)

A lightweight Python toolbox that learns low-order dynamical models from time-series data using SINDy (Sparse Identification of Nonlinear Dynamics), then reduces them into a compact modal form for fast simulation, control design, and interpretability.

Input: state or measurement time-series (and optionally control inputs)

Output: a sparse nonlinear ODE model 
𝑥˙ = 𝑓(𝑥,𝑢) x˙ = f(x,u) 
and a reduced modal representation capturing the dominant dynamics

Use cases: system identification, ROMs (reduced-order models), controller prototyping, digital twins

References:
  1. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).
     Discovering governing equations from data by sparse identification of nonlinear dynamical systems.
     Proceedings of the National Academy of Sciences (PNAS).
