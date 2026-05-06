# 🌌 Physical Simulations

This repository contains a collection of Python scripts simulating various physical phenomena. Each folder in the `src` directory corresponds to a specific physical problem, with detailed descriptions and theoretical background provided in the header of each file.

## 🧮 Implemented Simulations

* **diff_equations** - Solving various physical problems using numerical integration methods, including Euler, Verlet, and Leapfrog algorithms.
* **wator** - Implementation of the Wa-Tor model: a population dynamics simulation of a predator-prey system in a toroidal world.
* **Ising** - Simulation of the 2D Ising model, investigating the evolution of cluster sizes and the behavior of correlation functions over time.
* **SOC, GS and DLA** - A collection of advanced models, including: 
  * Self-Organized Criticality (SOC) through the Abelian sandpile model and its avalanche dynamics.
  * Solving the Gray-Scott reaction-diffusion equations.
  * Diffusion-Limited Aggregation (DLA) modeling fractal growth observed in various biological and physical systems.

## 🛠️ Technologies & Libraries
The simulations heavily rely on the scientific Python stack:
* **Python 3.x**
* **NumPy & SciPy** - For fast matrix operations, data fitting, and solving differential equations.
* **Matplotlib** - For data visualization and creating real-time simulation animations.
* **Numba** - Utilizing JIT (Just-In-Time) compilation to significantly accelerate computational bottlenecks and loop-heavy calculations.

## 🚀 How to Run

1. Clone this repository:
```bash
git clone [https://github.com/Mkasprzyk04/Symulacje-komputerowe-w-fizyce.git](https://github.com/Mkasprzyk04/Symulacje-komputerowe-w-fizyce.git)
cd Symulacje-komputerowe-w-fizyce
