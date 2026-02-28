---
layout: post
comments: true
title: "Training with sklearn, Deploying with the Right Runtime"
excerpt: "Use frameworks to learn. Use lean runtimes to serve."
date: 2026-02-28
category:
tags: machine-learning, mlops, deployment
---

## Compiling sklearn Trees to Source Code: Deploying ML with the Right Runtime

Deploying a machine learning model is not the same problem as training it.

Training environments are built for flexibility. They optimize for experimentation, rapid iteration, rich APIs, and numerical breadth. Deployment environments optimize for something very different: latency, memory footprint, startup time, packaging simplicity, and operational cost.

In this post I use decision trees and random forests as a controlled simplification of a broader issue: how we deploy ML models in production. Trees are useful here precisely because their inference logic is easy to reason about. They let us isolate the systems problem without hiding behind heavy linear algebra or GPU acceleration. The point is not that trees are special. The point is that deployment deserves its own engineering decisions.

TreeCompiler is a small proof of concept built around that idea. It compiles scikit-learn DecisionTreeClassifier and RandomForestClassifier models into standalone Python or Go source code. No sklearn. No numpy. No scientific stack at inference time.

The goal is not to replace frameworks. The goal is to make explicit a systems principle: inference deserves its own runtime.

---

## A Quick Reminder: How Trees and Forests Work

A decision tree recursively partitions the feature space using axis-aligned splits. At each internal node, a single feature and threshold are chosen to reduce impurity (for example using Gini or entropy). Inference is simply walking the tree from root to leaf by evaluating comparisons like `x[j] <= threshold`. A random forest is an ensemble of such trees trained on bootstrapped samples and random feature subsets. At inference time, each tree produces a prediction (or probability vector), and the forest aggregates them, typically by averaging probabilities (soft voting).

<div style="text-align: center;">
  <img src="https://miro.medium.com/v2/1*i0o8mjFfCn-uD79-F1Cqkw.png" alt="Decision tree and random forest diagram" />
</div>

If you want a deeper refresher, the scikit-learn documentation provides a clear overview of the algorithmic details: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

The important part for this discussion is this: inference is just branching logic and basic arithmetic. There is no heavy linear algebra involved.

---

## Training Runtime vs Inference Runtime

When we talk about inference cost, we usually think about compute: CPU cycles per forward pass, vectorized math, numerical throughput.

For tree-based models, the raw computation is trivial. A decision tree inference is a sequence of comparisons. A random forest repeats that process multiple times and averages results.

The expensive part is everything around that computation.

Importing large numerical libraries. Deserializing pickled objects. Allocating numpy arrays. Shipping container images with hundreds of megabytes of dependencies. Managing environment compatibility across Python versions and architectures.

Training environments are optimized for experimentation. Inference environments are optimized for stability, latency, and cost. Treating them as the same thing is convenient, but rarely optimal.

TreeCompiler explores what happens when we separate those concerns.

---

## What the Compiler Actually Does

Scikit-learn stores a trained decision tree as arrays: left children, right children, feature indices, thresholds, and class counts per node. Prediction consists of walking those arrays until a leaf is reached.

TreeCompiler extracts that structure into an intermediate representation and generates explicit source code with nested if/else branches.

For a small tree, the generated Python looks like this:

```python
def predict_proba(x):
    if x[2] <= 2.45:
        return [1.0, 0.0, 0.0]
    else:
        if x[3] <= 1.75:
            return [0.0, 0.91, 0.09]
        else:
            return [0.0, 0.02, 0.98]
```

No imports. No runtime dependencies. Just comparisons.

For a random forest, the compiler generates one function per tree and performs soft voting by averaging predicted probabilities.

Once inference becomes explicit source code, two structural changes happen.

First, sklearn is no longer required at runtime.

Second, the model stops being a serialized artifact and becomes part of the executable.

That shift directly impacts latency, memory usage, packaging, and deployment complexity.

---

## Benchmark Setup

To make the impact concrete, I benchmarked three deployment strategies on AWS Lambda, all using the same model:

RandomForestClassifier, 50 trees, depth 8, 4 classes, 20 features.

Each function was configured with 128MB of memory, which is the minimum allocation for AWS Lambda.

Deployment strategies:

1. sklearn inside a Docker image.
2. Compiled Python (generated source, standard Python runtime).
3. Compiled Go (generated source compiled into a static binary).

Python is an interpreted language. At runtime, the interpreter loads modules dynamically and executes bytecode. Go is a compiled language. The source is compiled ahead of time into a static binary that runs directly on the operating system without an interpreter.

Once inference is reduced to pure branching logic, the runtime itself becomes the dominant factor.

---

## Python vs Compiled Python (Same Language, Different Runtime Shape)

| Metric    | sklearn (Docker) | Python Compiled |
| --------- | ---------------- | --------------- |
| Cold init | 2.05s            | 450.8ms         |
| Warm p50  | 22.4ms           | 1.8ms           |
| Warm p95  | 30.5ms           | 15.7ms          |
| Memory    | 201 MB           | 127 MB          |

Cold init measures how long the execution environment takes to initialize before handling the first request.

Warm p50 is the median latency once the environment is already initialized.

Warm p95 captures tail latency, reflecting the slowest 5% of requests.

Memory is the peak memory consumption during execution.

Now the meaningful comparisons.

Cold initialization drops from 2.05 seconds to 450.8 milliseconds. That is roughly **4.5x faster**, or about a 78% reduction in startup time.

Warm p50 latency drops from 22.4ms to 1.8ms. That is **over 12x faster**, roughly a 92% reduction in median latency.

Warm p95 drops from 30.5ms to 15.7ms, almost **2x faster** at the tail.

Memory decreases from 201MB to 127MB, a **37% reduction**.

Nothing about the model changed. Only the runtime environment did. Even within Python, removing the training stack from the inference path produces structural gains.

---

## Compiled Python vs Compiled Go (Different Runtime Model)

| Metric    | Python Compiled | Go Compiled |
| --------- | --------------- | ----------- |
| Cold init | 450.8ms         | 78.3ms      |
| Warm p50  | 1.8ms           | 1.1ms       |
| Warm p95  | 15.7ms          | 11.5ms      |
| Memory    | 127 MB          | 21 MB       |

Cold initialization drops from 450.8ms to 78.3ms. That is roughly **5.7x faster**, an 83% reduction.

Warm p50 improves from 1.8ms to 1.1ms, about **1.6x faster**. The improvement is smaller because both implementations already execute minimal logic.

Warm p95 drops from 15.7ms to 11.5ms, improving tail latency by roughly **27%**.

Memory drops from 127MB to 21MB, an **83% reduction**, or about **6x smaller**.

The difference here is architectural. Python carries interpreter overhead even when executing simple branching code. Go compiles ahead of time into a static binary with minimal runtime initialization.

---

## Translating This Into Economic Impact

In serverless systems, cost is determined primarily by execution time and configured memory.

Reducing cold starts from 2.05 seconds to 78 milliseconds changes how scaling behaves under bursty traffic. Those seconds accumulate during scaling events.

Memory configuration also directly affects cost. AWS Lambda requires a minimum of 128MB, but higher memory tiers increase both price and allocated CPU share. A function that peaks at 21MB gives you confidence that 128MB is more than sufficient, whereas a function peaking near 200MB forces you into higher memory tiers for safety.

<div style="text-align: center;">
  <img src="https://github.com/barufa/barufa.github.io/blob/main/assets/img/treecompiler_benchmark.svg" alt="Benchmark" />
</div>

If you process millions of requests per day, a 12x reduction in median latency translates into dramatically lower billed compute time. An 83% reduction in memory usage reduces the need to overprovision memory for safety margins.

---

## Broader Lesson

This idea is not limited to trees.

In a [previous post](https://stuckinalocalminima.com/blog/2025/sklearn-faiss), I explored a similar separation between experimentation tooling and production inference when migrating PCA projections from scikit-learn to Faiss for scalable vector search. The theme was the same: use rich libraries for training and validation, but deploy inference using the runtime that best matches operational constraints. In that case, benchmarking showed roughly a **1.77× improvement** in throughput simply by moving the projection step to a runtime designed for high-performance inference.

Trees simply make that principle easier to visualize because their inference logic is explicit and finite.

The broader lesson is straightforward.

Inference is a systems problem as much as it is a modeling problem. The largest optimizations are not just in the math. They are in choosing the right runtime for the job.

---

## References

TreeCompiler repository: [https://github.com/barufa/TreeCompiler](https://github.com/barufa/TreeCompiler)

scikit-learn DecisionTreeClassifier documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

scikit-learn Tree module overview: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

Related post on separating training and inference runtimes: [https://stuckinalocalminima.com/blog/2025/sklearn-faiss](https://stuckinalocalminima.com/blog/2025/sklearn-faiss)
