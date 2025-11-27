# Multi-Agent Game Search: AI Warehouse Competition

Implementation of adversarial zero-sum game algorithms for autonomous warehouse robots competing to deliver packages efficiently.

## Problem Description

Two robots (R1 and R2) compete in a warehouse environment to deliver packages to destinations, earning points based on Manhattan distance.

**Game Environment:**
- 5×5 grid with 2 robots, 2 charging stations, 2 packages and 2 destinations
- Each robot starts with battery charge and no packages
- Available actions: Move (Up/Down/Left/Right), Pick Up, Drop Off, Charge
- Scoring: When a robot delivers a package, it earns `2 × Manhattan_distance` points
- Each action costs 1 battery unit
- Game ends when any robot runs out of battery OR max steps reached

---

## Algorithms Implemented

### 1. Improved Greedy Agent
- Single-step lookahead using a custom heuristic function
- Advantage: Fast computation
- Disadvantage: Myopic, does not anticipate opponent strategies

### 2. Resource-Bounded Minimax
- Game tree search with time-limited evaluation (1 second per move)
- Assumes optimal opponent play
- Exponential complexity O(b^d) but limited by depth

### 3. Alpha-Beta Pruning
- Minimax with branch pruning for efficiency
- Same move quality as Minimax, ~90% faster
- Best case complexity O(b^(d/2))

### 4. Expectimax Agent
- Probabilistic game tree search against imperfect opponents
- Models opponent with probability distribution
- Better against realistic (greedy) players

---

## Algorithm Complexity

| Algorithm | Time | Space | Optimality |
|-----------|------|-------|-----------|
| Greedy | O(b) | O(1) | Heuristic |
| Minimax | O(b^d) | O(bd) | Optimal |
| Alpha-Beta | O(b^(d/2)) | O(bd) | Optimal |
| Expectimax | O(b^d) | O(bd) | Optimal vs prob |

---

## Files

- `Agent.py` - Agent classes
- `WarehouseEnv.py` - Game environment
- `submission.py` - Main algorithms
- `main.py` - Game runner
- `assignment.pdf` - Full problem statement
- `report.pdf` - Complete solution

---

## Usage

```bash
python main.py greedy random -t 0.5 -c 200 --console_print
python main.py minimax alpha_beta -t 1.0 -c 200 --tournament 10
```

---

For detailed mathematical formulations, complexity analysis, and complete pseudocode:
...

See assignment.pdf and report.pdf for complete details.
