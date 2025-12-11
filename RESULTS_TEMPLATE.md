# Phase 3 Results Report Template

**Author:** Laiba Akram
**Student ID:** 42943
**Course:** BSCS-7A - Theory of Programming Languages
**Date:** [Fill in date]

---

## 1. Executive Summary

[Brief overview of findings - 2-3 paragraphs summarizing key results]

---

## 2. Dataset Statistics

### 2.1 Data Collection Summary

| Language   | Functions Collected | Average LOC | Total Files |
|------------|---------------------|-------------|-------------|
| Python     | [Fill in]          | [Fill in]   | [Fill in]   |
| Java       | [Fill in]          | [Fill in]   | [Fill in]   |
| JavaScript | [Fill in]          | [Fill in]   | [Fill in]   |
| Rust       | [Fill in]          | [Fill in]   | [Fill in]   |
| **Total**  | **[Fill in]**      | **[Fill in]**| **[Fill in]** |

### 2.2 Feature Statistics

**Total Features Extracted:** [Number]

**Feature Categories:**
- AST-based metrics: [Number]
- Complexity metrics: [Number]
- PL-level features: [Number]
- Concurrency features: [Number]

---

## 3. Clustering Results

### 3.1 Algorithm Performance Comparison

| Method       | Best Parameters | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|--------------|----------------|------------------|----------------|-------------------|
| K-Means      | k=[N]          | [Score]          | [Score]        | [Score]           |
| DBSCAN       | eps=[X], min_samples=[Y] | [Score]  | [Score]        | [Score]           |
| Hierarchical | k=[N], linkage=[type]    | [Score]  | [Score]        | [Score]           |

**Best Performing Method:** [Method name] with silhouette score of [Score]

### 3.2 Cluster Analysis

**Number of Clusters Found:** [Number]

**Cluster Composition:**

| Cluster | Size | Dominant Language | Python % | Java % | JavaScript % | Rust % |
|---------|------|-------------------|----------|--------|--------------|--------|
| 0       | [N]  | [Language]        | [%]      | [%]    | [%]          | [%]    |
| 1       | [N]  | [Language]        | [%]      | [%]    | [%]          | [%]    |
| 2       | [N]  | [Language]        | [%]      | [%]    | [%]          | [%]    |
| ...     | ...  | ...               | ...      | ...    | ...          | ...    |

### 3.3 Key Observations from Clustering

1. **Static vs Dynamic Typing:**
   - [Describe whether statically-typed languages (Java, Rust) cluster together]
   - [Provide cluster numbers and percentages]

2. **Memory Safety Patterns:**
   - [Describe clustering of memory-safe languages]
   - [Compare Rust clusters with Python/JavaScript/Java]

3. **Paradigm Influence:**
   - [Describe any paradigm-based clustering patterns]
   - [Functional vs OOP patterns]

4. **Unexpected Findings:**
   - [List any surprising clustering results]

### 3.4 Visualizations

- **Figure 1:** t-SNE visualization shows [describe what you see]
- **Figure 2:** PCA plot reveals [describe patterns]
- **Figure 3:** Cluster distribution indicates [describe composition]

---

## 4. Prediction Results

### 4.1 Model Performance Comparison

| Model         | Accuracy | Precision | Recall | F1-Score | Training Time |
|---------------|----------|-----------|--------|----------|---------------|
| Random Forest | [Score]  | [Score]   | [Score]| [Score]  | [Time]        |
| LSTM          | [Score]  | [Score]   | [Score]| [Score]  | [Time]        |
| Transformer*  | [Score]  | [Score]   | [Score]| [Score]  | [Time]        |

*If implemented

**Best Performing Model:** [Model name] with [Score]% accuracy

### 4.2 Confusion Matrix Analysis

**Random Forest Confusion Matrix:**

```
              Python  Java  JavaScript  Rust
Python        [TP]    [FP]  [FP]        [FP]
Java          [FN]    [TP]  [FP]        [FP]
JavaScript    [FN]    [FN]  [TP]        [FP]
Rust          [FN]    [FN]  [FN]        [TP]
```

**Key Misclassifications:**
- [Language A] often confused with [Language B]: [Explain why]
- [Pattern description]

### 4.3 Feature Importance

**Top 10 Most Important Features:**

| Rank | Feature                    | Importance Score |
|------|----------------------------|------------------|
| 1    | [Feature name]            | [Score]          |
| 2    | [Feature name]            | [Score]          |
| 3    | [Feature name]            | [Score]          |
| 4    | [Feature name]            | [Score]          |
| 5    | [Feature name]            | [Score]          |
| 6    | [Feature name]            | [Score]          |
| 7    | [Feature name]            | [Score]          |
| 8    | [Feature name]            | [Score]          |
| 9    | [Feature name]            | [Score]          |
| 10   | [Feature name]            | [Score]          |

**Insights:**
- [Describe which types of features matter most]
- [AST metrics vs traditional metrics]
- [PL-specific features]

---

## 5. Hypothesis Testing

### Original Hypothesis

> "For function-level tasks of equivalent functionality, LLM-generated code in statically typed, memory-safe languages (e.g., Rust, Go, Java) will exhibit higher syntax complexity (e.g., deeper ASTs, higher cyclomatic complexity) than generated code in dynamically typed languages (e.g., Python, JavaScript), and these complexity differences will correlate with summarization quality across languages."

### 5.1 Syntax Complexity by Language

**Average Complexity Metrics:**

| Language   | AST Depth | McCabe CC | Halstead Volume | LOC   |
|------------|-----------|-----------|-----------------|-------|
| Python     | [Avg]     | [Avg]     | [Avg]           | [Avg] |
| Java       | [Avg]     | [Avg]     | [Avg]           | [Avg] |
| JavaScript | [Avg]     | [Avg]     | [Avg]           | [Avg] |
| Rust       | [Avg]     | [Avg]     | [Avg]           | [Avg] |

**Statistical Significance:**
- [Describe statistical tests performed]
- [P-values and confidence intervals]

### 5.2 Hypothesis Validation

**Part 1: Complexity Differences**

☐ **SUPPORTED** / ☐ **NOT SUPPORTED** / ☐ **PARTIALLY SUPPORTED**

**Evidence:**
- [List evidence for/against]
- [Statistical test results]
- [Specific examples]

**Part 2: Clustering by PL Features**

☐ **SUPPORTED** / ☐ **NOT SUPPORTED** / ☐ **PARTIALLY SUPPORTED**

**Evidence:**
- [Do languages cluster by typing discipline?]
- [Do memory-safe languages group together?]
- [What drives clustering: syntax or semantics?]

---

## 6. Complexity Metric Analysis

### 6.1 McCabe Cyclomatic Complexity

**Distribution by Language:**
- Python: Mean=[X], Median=[Y], Std=[Z]
- Java: Mean=[X], Median=[Y], Std=[Z]
- JavaScript: Mean=[X], Median=[Y], Std=[Z]
- Rust: Mean=[X], Median=[Y], Std=[Z]

**Key Findings:**
- [Which language has highest complexity?]
- [Are statically-typed languages more complex?]

### 6.2 Halstead Metrics

**Volume Analysis:**
- [Describe Halstead volume patterns]
- [Correlation with language features]

**Difficulty & Effort:**
- [Which languages show higher difficulty?]
- [Implications for code understanding]

### 6.3 AST-Based Metrics

**AST Depth:**
- [Compare average depths across languages]
- [Relationship to language syntax]

**Branching Factor:**
- [Describe tree structure differences]
- [Language-specific patterns]

---

## 7. PL Feature Impact

### 7.1 Typing Discipline

**Static vs Dynamic Typing:**
- [Compare complexity metrics]
- [Clustering behavior]
- [Prediction accuracy]

**Findings:**
- [Do static languages cluster separately?]
- [Can models distinguish static from dynamic?]

### 7.2 Memory Model

**Ownership (Rust) vs Managed (Java, Python, JS):**
- [Unique patterns in Rust code]
- [Memory safety impact on complexity]

### 7.3 Concurrency Patterns

**Concurrency Usage:**
- Async/Await: [% of functions]
- Threading: [% of functions]
- Event Loop: [% of functions]

**Impact on Complexity:**
- [How concurrency affects metrics]
- [Language-specific concurrency patterns]

---

## 8. Visualizations Summary

### Key Figures

1. **t-SNE Clustering (Figure X)**
   - Shows: [Description]
   - Insight: [Key takeaway]

2. **Elbow Curve (Figure X)**
   - Optimal clusters: [Number]
   - Justification: [Reason]

3. **Feature Importance (Figure X)**
   - Top feature: [Name]
   - Implication: [Meaning]

4. **Confusion Matrix (Figure X)**
   - Best predictions: [Languages]
   - Worst predictions: [Languages]

5. **Metric Distributions (Figure X)**
   - Pattern: [Description]
   - Significance: [Meaning]

---

## 9. Discussion

### 9.1 Interpretation of Results

[2-3 paragraphs discussing what the results mean in the context of your hypothesis]

### 9.2 PL Theory Connections

**How do results relate to Programming Language Theory?**
- [Typing theory]
- [Semantics and syntax]
- [Language design principles]

### 9.3 Implications for Code Summarization

[Discuss how these findings affect code summarization and generation tools]

### 9.4 Limitations

1. **Dataset Size:** [Discuss limitations]
2. **Language Coverage:** [Missing languages]
3. **Code Domains:** [Type of code analyzed]
4. **Metric Limitations:** [AST parser limitations]
5. **Model Limitations:** [ML model constraints]

### 9.5 Threats to Validity

**Internal Validity:**
- [Confounding factors]
- [Control measures]

**External Validity:**
- [Generalizability]
- [Representativeness of dataset]

---

## 10. Conclusions

### 10.1 Summary of Findings

1. [Key finding 1]
2. [Key finding 2]
3. [Key finding 3]
4. [Key finding 4]
5. [Key finding 5]

### 10.2 Hypothesis Verdict

**Overall Assessment:** ☐ SUPPORTED / ☐ NOT SUPPORTED / ☐ PARTIALLY SUPPORTED

**Justification:**
[Explain the verdict based on all evidence]

### 10.3 Contributions

This research contributes:
1. [Contribution to PL theory]
2. [Contribution to code analysis]
3. [Contribution to ML for code]

---

## 11. Future Work

### Immediate Extensions

1. **Additional Languages:** [Go, C++, Swift, etc.]
2. **Larger Dataset:** [Scale to 10K+ functions per language]
3. **Code Summarization:** [Implement actual summarization task]

### Research Directions

1. [Direction 1]
2. [Direction 2]
3. [Direction 3]

### Methodological Improvements

1. [Better feature engineering]
2. [Advanced ML models]
3. [Cross-language transfer learning]

---

## 12. References

[List all papers, tools, and datasets used]

1. Phase 01 document
2. Phase 02 document
3. [Additional references]

---

## Appendix A: Detailed Tables

[Include detailed statistical tables]

## Appendix B: Code Samples

[Include representative code samples from each cluster]

## Appendix C: Additional Visualizations

[Include supplementary figures]

---

**End of Report**
