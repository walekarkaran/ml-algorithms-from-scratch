# Machine Learning Algorithms from Scratch

Classic machine learning algorithms implemented from first principles in **pure NumPy**, one Jupyter notebook per algorithm.
No scikit-learn and no SciPy; the only plotting library is Matplotlib.

Each notebook follows the same structure:

1. **Theory** — the idea, the maths and the algorithm steps
2. **Implementation** — a clean, commented class or function written from scratch
3. **Experiments** — demos on generated datasets with visualisations (decision boundaries, loss curves, dendrograms, …)
4. **Summary** — strengths, weaknesses and when to use the method

## Contents

### Regression
| # | Notebook | Highlights |
|---|---|---|
| 01 | [Linear Regression](01_Linear_Regression.ipynb) | Normal equation vs gradient descent, learning-rate study, multiple regression |
| 02 | [Polynomial Regression](02_Polynomial_Regression.ipynb) | Feature expansion, bias–variance trade-off, ridge regularisation, 2-D surface fit |
| 03 | [Logistic Regression](03_Logistic_Regression.ipynb) | Sigmoid + cross-entropy, decision boundary, softmax (multiclass) regression |

### Classification
| # | Notebook | Highlights |
|---|---|---|
| 04 | [Decision Tree — ID3](04_Decision_Tree_ID3.ipynb) | Entropy, information gain, Play Tennis dataset, tree visualisation |
| 05 | [Decision Tree — C4.5](05_Decision_Tree_C45.ipynb) | Gain ratio, continuous attributes, pessimistic (binomial) error pruning |
| 06 | [Decision Tree — CART](06_Decision_Tree_CART.ipynb) | Gini / MSE, vectorised split search, cost-complexity pruning, regression trees |
| 07 | [K-Nearest Neighbours](07_KNN.ipynb) | Minkowski distances, distance-weighted voting, choosing *k*, feature scaling, KNN regression |
| 08 | [Support Vector Machine](08_SVM.ipynb) | Hinge-loss sub-gradient descent, SMO solver for the dual, linear / polynomial / RBF kernels |
| 09 | [Naive Bayes](09_Naive_Bayes.ipynb) | Gaussian, Multinomial (spam filter) and Categorical NB, Laplace smoothing |

### Evaluation Metrics
| # | Notebook | Highlights |
|---|---|---|
| 10 | [Evaluation Metrics](10_Evaluation_Metrics.ipynb) | Accuracy, precision, recall, F1, confusion matrix, classification report, ROC curve, AUC |

### Ensemble Methods
| # | Notebook | Highlights |
|---|---|---|
| 11 | [Bagging & Random Forest](11_Bagging_Random_Forest.ipynb) | Bootstrap sampling, out-of-bag score, feature importance, tree decorrelation |
| 12 | [AdaBoost](12_AdaBoost.ipynb) | Weighted decision stumps, sample-weight evolution, training-error bound |
| 13 | [Gradient Boosting](13_Gradient_Boosting.ipynb) | Residual fitting (regression), log-loss with Newton leaf values (classification), shrinkage |

### Clustering
| # | Notebook | Highlights |
|---|---|---|
| 14 | [Clustering — Introduction & Types](14_Clustering_Introduction.ipynb) | Partitioning / hierarchical / density / model / grid-based, distances, silhouette, purity, Rand index |
| 15 | [K-Means](15_KMeans.ipynb) | Lloyd's algorithm, K-Means++, elbow & silhouette, colour quantisation |
| 16 | [Hierarchical Clustering](16_Hierarchical_Clustering.ipynb) | Single / complete / average / Ward linkage (Lance–Williams), hand-drawn dendrograms, bisecting K-Means |
| 17 | [DBSCAN](17_DBSCAN.ipynb) | Core / border / noise points, k-distance plot for `eps`, comparison with K-Means |

## Getting started

```bash
git clone https://github.com/walekarkaran/ml-algorithms-from-scratch.git
cd ml-algorithms-from-scratch
pip install -r requirements.txt
jupyter notebook
```

Every notebook is self-contained: datasets are either generated in code (blobs, moons, circles, …) or small
classic tables written inline (e.g. *Play Tennis*), so there is nothing to download. Each notebook runs in a few seconds.

## Correctness checks

During development the implementations were cross-checked against reference libraries (the notebooks themselves do not use them):

- **Metrics** — confusion matrix, precision, recall, F1 and ROC-AUC match `sklearn.metrics` exactly, including tied scores
- **CART** — identical predictions to `sklearn.tree.DecisionTreeClassifier` (depth 4)
- **Hierarchical clustering** — merge heights and flat clusters match `scipy.cluster.hierarchy.linkage` for all four linkages
- **DBSCAN** — identical clusters and noise points to `sklearn.cluster.DBSCAN`
- **K-Means** — same inertia and silhouette score as `sklearn.cluster.KMeans`
- **ID3, C4.5, Naive Bayes** — reproduce the textbook *Play Tennis* results (Quinlan, Mitchell)

## Requirements

- Python 3.9+
- NumPy, Matplotlib, Jupyter (see [`requirements.txt`](requirements.txt))

## License

Released under the [MIT License](LICENSE).
