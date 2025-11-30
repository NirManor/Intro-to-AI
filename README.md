# Introduction to Artificial Intelligence - Course Projects

Complete implementations of fundamental AI algorithms from an introductory artificial intelligence course, covering adversarial game search and machine learning techniques.

## Course Information

- **Course Number:** 02360501
- **Course Name:** Introduction to Artificial Intelligence
- **Institution:** Technion - Israel Institute of Technology
- **Faculty:** Computer Science
- **Semester:** Spring 2023
- **Language:** Python 3

---

## Repository Overview

This repository contains practical implementations of core AI concepts with a focus on:
- Adversarial game playing algorithms
- Decision tree learning
- Markov Decision Processes (MDPs)

All implementations emphasize algorithmic understanding, mathematical rigor, and practical application to real-world problems.

---

## Projects

### 1. Multi-Agent Game Search (HW2)
**Path:** `Multi-Agent-Game-Search/`

**Topic:** Adversarial zero-sum games and game-playing algorithms

**Algorithms Implemented:**
- **Improved Greedy:** Fast heuristic-based single-step lookahead
- **Minimax:** Optimal game tree search assuming perfect opponent play
- **Alpha-Beta Pruning:** Efficient variant of Minimax with branch pruning
- **Expectimax:** Probabilistic opponent modeling for realistic scenarios

**Application:** Autonomous warehouse robots competing to deliver packages efficiently, earning points based on Manhattan distance.

**Key Concepts:**
- Game tree representation
- Heuristic evaluation functions
- Minimax with time limits
- Alpha-beta pruning optimization
- Branching factor analysis

**Files:**
- `Agent.py` - Agent classes
- `WarehouseEnv.py` - 5×5 grid game environment
- `submission.py` - Game algorithms
- `main.py` - Game runner and visualization
- `README.md` - Detailed documentation
- `assignment.pdf` - Problem specification
- `report.pdf` - Solution report

---

### 2. Machine Learning & MDPs (HW3 - WET)
**Path:** `Machine-Learning-and-MDPs/`

**Topics:** Decision tree learning and sequential decision-making

**Algorithms Implemented:**

**Part A: Decision Tree Learning (ID3)**
- Information gain-based feature selection
- Continuous feature discretization via dynamic thresholding
- Pre-pruning for generalization
- k-fold cross-validation for parameter tuning

**Part B: Markov Decision Processes**
- Value Iteration algorithm
- Policy Iteration algorithm
- Bellman equations and optimality principle
- Policy extraction from value functions

**Applications:**
- **ID3:** Medical tumor diagnosis (benign vs. malignant) with 30 continuous features
- **MDPs:** Grid world navigation with stochastic transitions

**Key Concepts:**
- Entropy and information gain
- Continuous feature discretization
- Overfitting and regularization
- Markov property and state abstraction
- Bellman optimality equations
- Value vs. policy iteration tradeoffs

**Files:**
- `utils.py` - Data loading, accuracy metrics
- `ID3.py` - Decision tree algorithm
- `ID3_experiments.py` - Experiments and cross-validation
- `mdp_rl_implementation.py` - MDP solvers
- `README.md` - Detailed documentation
- `assignment.pdf` - Problem specification
- `report.pdf` - Solution report

---

## Repository Structure

```
Intro-to-AI/
├── README.md (this file)
│
├── Multi-Agent-Game-Search/
│   ├── README.md
│   ├── Agent.py
│   ├── WarehouseEnv.py
│   ├── submission.py
│   ├── main.py
│   ├── assignment.pdf
│   └── report.pdf
│
└── Machine-Learning-and-MDPs/
    ├── README.md
    ├── utils.py
    ├── ID3.py
    ├── ID3_experiments.py
    ├── mdp_rl_implementation.py
    ├── assignment.pdf
    └── report.pdf
```

---

## Getting Started

### Prerequisites
- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for cross-validation)

### Installation
```bash
git clone https://github.com/NirManor/Intro-to-AI.git
cd Intro-to-AI
pip install numpy matplotlib scikit-learn
```

### Running the Projects

**Game Search:**
```bash
cd Multi-Agent-Game-Search
python main.py greedy random -t 0.5 -c 200 --console_print
python main.py minimax alpha_beta -t 1.0 --tournament 10
```

**Machine Learning & MDPs:**
```bash
cd Machine-Learning-and-MDPs
python -c "from ID3_experiments import basic_experiment; basic_experiment()"
```

---

## Algorithm Complexity Summary

### Game Search Algorithms

| Algorithm | Time | Space | Optimality |
|-----------|------|-------|-----------|
| Greedy | O(b) | O(1) | Heuristic |
| Minimax | O(b^d) | O(bd) | Optimal |
| Alpha-Beta | O(b^(d/2)) | O(bd) | Optimal |
| Expectimax | O(b^d) | O(bd) | Probabilistic |

### Learning Algorithms

| Algorithm | Time | Test | Space |
|-----------|------|------|-------|
| ID3 Tree | O(n log n) | O(log n) | O(n) |
| Value Iteration | O(S²A × iter) | O(S) | O(S) |
| Policy Iteration | O(S²A + S³) | O(S) | O(S) |

---

## Key Features

### Game Search
- Custom heuristics combining score, battery, and opportunity
- Time-limited decision making (1 second per move)
- Stochastic transitions (80% success + 10% perpendicular)
- Handles large branching factors with pruning

### Machine Learning
- Dynamic discretization for continuous features
- Entropy-based splitting with information gain
- Cross-validation for hyperparameter tuning
- Action-reward MDP formulation

---

## Results & Performance

- **Game Search:** Alpha-Beta 10× faster than Minimax with same move quality
- **Decision Trees:** 95%+ accuracy on medical diagnosis task
- **MDPs:** Value Iteration converges in ~50 iterations on grid world

---

## Topics Covered

**Foundations:**
- Problem solving via search
- State space representation
- Heuristic evaluation

**Game Playing:**
- Adversarial search
- Zero-sum games
- Game tree exploration
- Opponent modeling

**Machine Learning:**
- Supervised learning via decision trees
- Feature selection and information theory
- Overfitting and pruning

**Optimization:**
- Markov Decision Processes
- Dynamic programming
- Bellman equations
- Value and policy functions

---

## References

1. Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.) - Chapter 5
2. Quinlan, J. R. (1986). "Induction of Decision Trees"
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
4. Bellman, R. E. (1957). "Dynamic Programming"

---

## License

Educational use. Course materials property of Technion.

---

**Repository:** https://github.com/NirManor/Intro-to-AI  
**Last Updated:** 2025  
**Status:** Complete implementations with comprehensive documentation
