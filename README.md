# Custom Machine Learning Core: From-Scratch Estimators

A pedagogical, low-level data science repository featuring three foundational classification algorithms built entirely from scratch using standard Python libraries and NumPy. This project bypasses high-level abstractions to expose the explicit mathematical logic, optimization strategies, and underlying mechanics of supervised learning pipelines.

---

## Core Architecture Philosophy

This repository is built for **architectural transparency, educational depth, and structural optimization**. By avoiding complex third-party frameworks, every estimator exposes raw matrix manipulation, probability accumulation, and information distributions.

* **Zero-Magic Implementations:** Built using native Python data structures and optimized conditional routing loops to cleanly isolate runtime bottlenecks.
* **Low-Resource Engineering:** Highly performant when running on constrained hardware (e.g., lightweight environments with dual-core processors and limited RAM) through single-pass vector collections and lazy training parameters.
* **Algorithmic Hardening:** Explicitly resolves real-world operational hazards, including floating-point underflow via log-space arithmetic and zero-frequency tracking via Laplace smoothing.

---

## Algorithmic Blueprints

### 1. Naive Bayes Classifier
A probabilistic model leveraging Bayes' Theorem with strong (naive) conditional independence assumptions. This engine evaluates categorical feature spaces via a multivariate **Bernoulli tracking pipeline** optimizing binary token flags.

#### 🧮 Mathematical Form
To protect calculations from severe floating-point underflow when continuous fractions drop below machine precision thresholds, posterior calculations are processed exclusively in the log-likelihood domain:

$$\log P(C_k \mid \mathbf{x}) \propto \log P(C_k) + \sum_{i=1}^{n} \left[ x_i \log \theta_{ki} + (1 - x_i) \log (1 - \theta_{ki}) \right]$$

Where:
* $\theta_{ki}$ represents the Laplace-smoothed conditional distribution parameter ($k=0.5$).
* The $(1 - x_i)$ vector tracks explicit penalties for vocabulary words that are **absent** from an incoming record.

---

### 2. K-Nearest Neighbors (KNN)
An instance-based, non-parametric spatial classification model that groups unclassified attributes by computing multi-dimensional geometric distances against an existing structural feature topology.

#### 🧮 Mathematical Form
Spatial projections are evaluated using clean Euclidean distance mappings across a continuous vector field:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

#### 🔧 Operational Attributes
* **Lazy Initialization:** The `.train()` configuration implements an $O(1)$ reference mapping layout, storing input points directly without upfront computational overhead.
* **Deterministic Tie-Breaking:** Employs sorted weight-frequency reductions to gracefully resolve ambiguous classification boundaries when neighbor vote counts are perfectly balanced.

---

### 3. ID3 Decision Tree Classifier
A greedy, top-down recursive partitioning algorithm that evaluates categorical feature arrays by isolating maximize informational value at each branch node.

#### 🧮 Mathematical Form
The split optimization engine evaluates systemic impurity using **Shannon Entropy**:

$$H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i$$

Node splits are dynamically prioritized by selecting attributes that maximize **Information Gain ($IG$)**, driving the child subsets toward absolute thermodynamic purity:

$$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

```text
               [ Attribute: Max Information Gain ]
                         /        \
                    (Value A)   (Value B)
                      /              \
         [ Leaf Node: Class 1 ]     [ Sub-Tree Recursive Split ]