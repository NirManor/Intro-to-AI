# Machine Learning & Markov Decision Processes (MDPs)

Implementation of decision tree learning (ID3 algorithm) and Markov Decision Process solvers (Value Iteration, Policy Iteration).

## Part A: Decision Tree Learning (ID3 Algorithm)

### Problem Description

**Dataset:** Medical imaging data with 30 continuous features for tumor classification
- **Target:** Diagnose benign (B) vs. malignant (M) tumors
- **Task:** Build decision tree classifier using ID3 algorithm

### Information Theory Foundations

**Entropy:**
```
H(S) = -∑(p_i × log₂(p_i))
```

**Information Gain:**
```
IG(S, A) = H(S) - ∑((|S_v| / |S|) × H(S_v))
```

### ID3 Algorithm

1. If all examples have same class: return Leaf
2. If features empty or stopping criteria met: return Leaf
3. Find best feature using maximum information gain
4. Split dataset by best feature's threshold
5. Recursively build subtrees

### Continuous Feature Discretization

For continuous features, dynamically find optimal threshold:
```
For each feature f:
  Sort unique values
  For each pair (v_i, v_{i+1}):
    Try threshold t = (v_i + v_{i+1}) / 2
    Calculate IG(split on t)
  Use threshold with maximum IG
```

### Pre-Pruning

Stop splitting when samples ≤ min_for_pruning parameter:
- Small m (m=1): Full tree, overfitting
- Medium m (m=50): Good generalization
- Large m (m=500): Underfitting

### Cross-Validation

Find optimal pruning parameter:
```
For m in parameter_values:
  5-fold cross-validation on training set
  Report average accuracy
Select m with highest accuracy
```

---

## Part B: Markov Decision Processes (MDPs)

### MDP Formulation

**Tuple:** ⟨S, A, P, R, γ⟩
- **S:** State space
- **A:** Action space
- **P:** Transition function P(s'|s, a)
- **R:** Reward function
- **γ:** Discount factor (0 < γ < 1)

### Bellman Equation

**Expected Utility:**
```
U(s) = R(s) + γ × ∑_{s'} P(s'|s,a) × U(s')
```

**Bellman Optimality:**
```
V*(s) = max_a [R(s) + γ × ∑_{s'} P(s'|s,a) × V*(s')]
```

### Value Iteration

```
1. Initialize V(s) = 0 for all states
2. Repeat until convergence:
     For each state s:
       V(s) ← max_a [R(s) + γ × ∑_{s'} P(s'|s,a) × V(s')]
3. Extract policy:
     π(s) = argmax_a [R(s) + γ × ∑_{s'} P(s'|s,a) × V(s')]
```

### Policy Iteration

```
1. Initialize policy π randomly
2. Repeat until convergence:
     Policy Evaluation:
       Compute V^π(s) until convergence
     Policy Improvement:
       π_new(s) ← argmax_a [R(s) + γ × ∑_{s'} P(s'|s,a) × V^π(s')]
     If π_new = π: done
     Else: π ← π_new
```

### Action-Reward Formulation

Traditional: R(s)
This assignment: R(s, a)

Modified Bellman:
```
V*(s) = max_a [R(s,a) + γ × ∑_{s'} P(s'|s,a) × V*(s')]
```

---

## Application Example: Grid World

**Environment:** 4×4 grid with walls, terminal states
- **States:** 11 (grid - walls)
- **Actions:** {Up, Down, Left, Right}
- **Transitions:** 0.8 succeed, 0.1 each perpendicular
- **Rewards:** R((2,4)) = -1, R((3,4)) = +1
- **Gamma:** 0.9

**Result:** Optimal policy navigates toward +1 goal, avoids -1

---

## Key Concepts

**Markov Property:** Future independent of past given present

**Policy:** Complete mapping of states to actions

**Value Function:** Expected utility from state following policy

**Optimal Policy:** Policy achieving maximum expected value

**Discount Factor:** Balance present vs. future rewards
- γ = 0: Only immediate (myopic)
- γ = 1: All future equal (only for guaranteed finite paths)
- 0.5 < γ < 0.99: Typical range

---

## Files

### ID3 Decision Trees
- `utils.py` - Data loading, accuracy metrics
- `ID3.py` - ID3 algorithm with pre-pruning
- `ID3_experiments.py` - Experiments and cross-validation

### MDPs
- `mdp_implementation.py` - Value Iteration, Policy Iteration

### Documentation
- `assignment.pdf` - Complete specification (100 points total)
- `report.pdf` - Solutions and results

---

## Complexity Comparison

| Algorithm | Training | Testing | Space |
|-----------|----------|---------|-------|
| ID3 Tree | O(n log n) | O(log n) | O(n) |
| Value Iteration | O(S²A × iter) | O(S) | O(S) |
| Policy Iteration | O(S²A + S³) | O(S) | O(S) |

---

## References

1. **Decision Trees:**
   - Quinlan, J. R. (1986). "Induction of Decision Trees"
   - Information gain, ID3 algorithm

2. **MDPs & Dynamic Programming:**
   - Bellman, R. E. (1957). "Dynamic Programming"
   - Sutton & Barto (2018). "Reinforcement Learning: An Introduction"

3. **Information Theory:**
   - Shannon, C. E. (1948). "A Mathematical Theory of Communication"

---

For complete mathematical formulations, pseudocode, and experimental results:
See `assignment.pdf` and `report.pdf` in this directory.

**Course:** Introduction to AI - Spring 2023
