# Introduction to Artificial Intelligence - Course Projects

Complete implementations of fundamental AI algorithms from an introduction to AI course, covering adversarial game search and machine learning techniques.

## Repository Overview

This repository contains practical implementations of core AI concepts learned in an introductory AI course, with a focus on:
- Adversarial game playing algorithms
- Decision tree learning
- Markov Decision Processes (MDPs)

All implementations emphasize algorithmic understanding, mathematical rigor, and practical application to real-world problems.

---

## Projects

### 1. Multi-Agent Game Search
**Path:** `Multi-Agent-Game-Search/`

**Topic:** Adversarial zero-sum games and game-playing algorithms

**Algorithms Implemented:**
- **Improved Greedy:** Fast heuristic-based single-step lookahead
- **Minimax:** Optimal game tree search assuming perfect opponent play
- **Alpha-Beta Pruning:** Efficient variant of Minimax with branch pruning
- **Expectimax:** Probabilistic opponent modeling for realistic scenarios

**Application:** Autonomous warehouse robots competing to deliver packages efficiently, earning points based on distance.

### 2. Machine Learning & MDPs
**Path:** `Machine-Learning-and-MDPs/`

**Topics:** Decision tree learning and sequential decision-making

**Algorithms Implemented:**
- **ID3 Decision Trees:** Information gain-based feature selection with pre-pruning
- **Value Iteration:** Dynamic programming for optimal value functions
- **Policy Iteration:** Direct policy optimization

**Applications:**
- Medical tumor diagnosis (benign vs. malignant)
- Grid world navigation with stochastic transitions

---

## Key Files

### Multi-Agent Game Search
- `Agent.py` - Agent implementations
- `WarehouseEnv.py` - Game environment
- `submission.py` - Main algorithms
- `main.py` - Game runner
- `README.md` - Detailed algorithm documentation
- `assignment.pdf` - Problem specification
- `report.pdf` - Solution report

### Machine Learning & MDPs
- `utils.py` - Utilities and data loading
- `ID3.py` - Decision tree algorithm
- `ID3_experiments.py` - Experiments and cross-validation
- `mdp_rl_implementation.py` - MDP solvers
- `README.md` - Detailed algorithm documentation
- `assignment.pdf` - Problem specification
- `report.pdf` - Solution report

---

## Getting Started

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn
```

### Running Examples
```bash
# Game search
cd Multi-Agent-Game-Search
python main.py greedy random -t 0.5 -c 200

# Decision trees & MDPs
cd Machine-Learning-and-MDPs
python -c "from ID3_experiments import basic_experiment; basic_experiment()"
```

---

## Course Information

- **Institution:** Technion - Israel Institute of Technology
- **Course:** Introduction to Artificial Intelligence
- **Semester:** Spring 2023
- **Topics:** Game search, decision trees, MDPs, dynamic programming

---

## Detailed Documentation

Each project folder contains a comprehensive README with:
- Problem description and formal definitions
- Algorithm explanations with pseudocode
- Complexity analysis
- Implementation details
- Examples and usage

See `Multi-Agent-Game-Search/README.md` and `Machine-Learning-and-MDPs/README.md` for complete documentation.
