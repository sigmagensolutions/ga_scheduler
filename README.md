# 🧬 Genetic Algorithm Workforce Scheduler

This project implements a genetic algorithm to evolve workforce schedules that best match a minute-by-minute demand curve. It supports synthetic and real-world demand data (e.g., from call center logs) and visually tracks capacity matching and fitness progression over time.

## 🔍 Problem Overview

Given:
- A demand curve (minutes of worker time needed per minute of the week)
- Worker constraints (full-time, consistent schedules, required days off)

The algorithm evolves schedules to minimize the sum of squared deviations between **demand** and **capacity**.

---

## 🛠️ Features

- ✅ Realistic workforce modeling (shift rules, consecutive days off, etc.)
- ✅ Support for real-world call center data (`call_data.csv`)
- ✅ Visual animation of the evolutionary process
- ✅ Fitness progression plot
- ✅ CSV export of best solution per generation
- ✅ Easily tunable parameters: population size, mutation rate, min/max workers, etc.

---

## 📊 Visualization

Top plot: Capacity (blue) vs Demand (black)  
Bottom plot: Fitness score over generations (lower = better)  
Gray dashed line: Best solution from the initial random population

---

## 📁 File Structure

- `evolutionary_scheduler.py` – Full code for demand ingestion, evolution, plotting
- `call_data.csv` – Realistic call center demand input (from Kaggle or other source)
- `best_solutions_by_generation.csv` – Output of best workforce per generation

---

## ▶️ How to Run

### 1. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate


## Install dependencies
pip install -r requirements.txt


## Run the scheduler
python evolutionary_scheduler.py


## Dependencies
numpy
matplotlib
pandas

## Example Output
You’ll see:
- Animated charts evolve over time
- Console output of the best workforce
- CSV with full historical bests

## Possible Future To Do List
- Add support for part-time workers
- Implement breaks and lunch periods
- Add GUI or web interface for parameter tuning