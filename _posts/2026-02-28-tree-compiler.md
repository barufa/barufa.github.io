---
layout: post
comments: true
title: "Training with scikit-learn, Deploying with the Right Runtime"
excerpt: "Use frameworks to learn. Use lean runtimes to serve."
date: 2026-02-28
category:
tags: [python, machinelearning, performance, mlops]
---

## Compiling scikit-learn Trees to Source Code: Deploying ML with the Right Runtime

Deploying a machine learning model is not the same problem as training it.

Training environments are built for flexibility. They optimize for experimentation, rapid iteration, rich APIs, and numerical breadth. Deployment environments optimize for something very different: latency, memory footprint, startup time, packaging simplicity, and operational cost.

In this post I use decision trees and random forests as a controlled simplification of a broader issue: how we deploy ML models in production. Trees are useful here precisely because their inference logic is easy to reason about. They let us isolate the systems problem without hiding behind heavy linear algebra or GPU acceleration. The point is not that trees are special. The point is that deployment deserves its own engineering decisions.

TreeCompiler is a small proof of concept built around that idea. It compiles scikit-learn DecisionTreeClassifier and RandomForestClassifier models into standalone Python or Go source code. No scikit-learn. No numpy. No scientific stack at inference time.

The goal is not to replace frameworks. The goal is to make explicit a systems principle: inference deserves its own runtime.

---

## A Quick Reminder: How Trees and Forests Work

A decision tree recursively partitions the feature space using axis-aligned splits. At each internal node, a single feature and threshold are chosen to reduce impurity (for example using Gini or entropy). Inference is simply walking the tree from root to leaf by evaluating comparisons like `x[j] <= threshold`. A random forest is an ensemble of such trees trained on bootstrapped samples and random feature subsets. At inference time, each tree produces a prediction (or probability vector), and the forest aggregates them, typically by averaging probabilities (soft voting).

<div style="text-align: center;">
  <img src="https://miro.medium.com/v2/1*i0o8mjFfCn-uD79-F1Cqkw.png" alt="Decision tree and random forest diagram" style="max-width: 100%; height: auto;" />
</div>

If you want a deeper refresher, the scikit-learn documentation provides a clear overview of the algorithmic details: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

The important part for this discussion is this: inference is just branching logic and basic arithmetic. There is no heavy linear algebra involved.

---

## Training Runtime vs Inference Runtime

When we talk about inference cost, we usually think about raw compute: CPU cycles per forward pass, vectorized math, and numerical throughput. For tree-based models, however, the computation itself is small. A decision tree prediction is mostly a sequence of comparisons, and a random forest repeats that process across multiple trees before averaging the results. That means the largest cost often comes from the machinery around the model rather than from the model logic itself: importing large numerical libraries, deserializing pickled objects, allocating arrays, shipping container images with hundreds of megabytes of dependencies, and managing compatibility across Python versions and architectures.

Those costs make sense in a training environment, where flexibility, experimentation, and rich APIs matter. They make much less sense in an inference environment, where the model has already been trained and the runtime only needs to execute a fixed set of operations. TreeCompiler explores what happens when we separate those concerns. By turning the learned tree structure into explicit source code, inference no longer depends on a general-purpose ML framework to load and execute a serialized artifact. The deployed program contains only the branching logic required for prediction, which reduces startup work, dependency loading, object allocation, and packaging overhead. In compiled targets, it also gives the compiler a simpler program to optimize: deterministic control flow, constant thresholds, direct comparisons, and small functions instead of a dynamic model object interpreted through a training framework.

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

The generated code has no imports and no runtime dependencies, only comparisons and simple arithmetic. For a random forest, TreeCompiler generates one function per tree and performs soft voting by averaging the predicted probabilities. Once inference is represented as explicit source code, the deployment shape changes: scikit-learn is no longer required at runtime, and the model is no longer a serialized artifact loaded by an external framework, but part of the executable itself. That shift directly affects latency, memory usage, packaging, and deployment complexity.

---

## Benchmark Setup

To make the impact concrete, I benchmarked three deployment strategies on AWS Lambda, all using the same model:

RandomForestClassifier, 50 trees, depth 8, 4 classes, 20 features.

Each function was configured with 128MB of memory, which is the minimum allocation for AWS Lambda.

Deployment strategies:

1. scikit-learn inside a Docker image.
2. Python code (generated source, standard Python runtime).
3. Compiled Go (generated source compiled into a static binary).

Python and Go expose two different optimization surfaces. In Python, the generated tree code is still executed by an interpreter, so the runtime sees the model as bytecode to evaluate step by step. In Go, the generated source is compiled ahead of time into native machine code. That gives the compiler a chance to apply optimizations before the function ever runs: lowering comparisons and arithmetic into direct CPU instructions, choosing efficient register and stack layouts.

Once inference is reduced to pure branching logic, the runtime itself becomes the dominant factor.

---

## Python vs Compiled Python (Same Language, Different Runtime Shape)

| Metric    | scikit-learn (Docker) | Python Compiled |
| --------- | --------------------- | --------------- |
| Cold init | 2.05s                 | 450.8ms         |
| Warm p50  | 22.4ms                | 1.8ms           |
| Warm p95  | 30.5ms                | 15.7ms          |
| Memory    | 201 MB                | 127 MB          |

Cold init measures the time required to initialize the execution environment before the first request. Warm p50 and warm p95 measure latency after the environment is already running: p50 captures the median request, while p95 captures tail latency near the slowest 5% of requests. Memory is the peak memory consumption observed during execution. Under those definitions, compiled Python changes the shape of the service even though the model is the same: cold initialization drops from 2.05 seconds to 450.8 milliseconds, roughly **4.5x faster**, or about a 78% reduction in startup time. Warm p50 falls from 22.4ms to 1.8ms, **over 12x faster**, roughly a 92% reduction in median latency. Warm p95 drops from 30.5ms to 15.7ms, almost **2x faster** at the tail. Memory decreases from 201MB to 127MB, a **37% reduction**.

Those improvements do not come from changing the forest, the features, or the prediction rule. They come from changing what has to exist at inference time. The generated Python function still runs on the Python interpreter, but it no longer needs to import and initialize scikit-learn, load a pickled object graph, allocate NumPy structures, or carry a full scientific stack just to execute a sequence of branches. Even within the same language, replacing a framework-driven runtime with explicit prediction code produces structural gains in startup time, latency, and memory usage.

---

## Compiled Python vs Compiled Go (Different Runtime Model)

| Metric    | Python Compiled | Go Compiled |
| --------- | --------------- | ----------- |
| Cold init | 450.8ms         | 78.3ms      |
| Warm p50  | 1.8ms           | 1.1ms       |
| Warm p95  | 15.7ms          | 11.5ms      |
| Memory    | 127 MB          | 21 MB       |

Moving from pure Python to compiled Go pushes the same generated model into a different execution model. Cold initialization drops from 450.8ms to 78.3ms, roughly **5.7x faster**, or an 83% reduction. Warm p50 improves from 1.8ms to 1.1ms, about **1.6x faster**, while warm p95 drops from 15.7ms to 11.5ms, improving tail latency by roughly **27%**. Memory falls from 127MB to 21MB, an **83% reduction**, or about **6x smaller**. The warm latency gains are smaller than the scikit-learn to compiled Python jump because both versions already execute minimal prediction logic. The remaining difference comes from the runtime around that logic.

In the Python version, the generated code is explicit, but it still runs through the Python interpreter and Python object model. In the Go version, the same tree structure is compiled ahead of time into a static binary, so the executable can start with much less runtime machinery and the compiler can lower the generated comparisons, constants, and function calls into native code. For a workload made almost entirely of deterministic branches and simple arithmetic, that matters: there is little numerical work left to optimize, so startup cost, memory footprint, and runtime dispatch become the dominant factors.

---

## Translating This Into Economic Impact

In serverless systems, latency and cost are tied to the same underlying factors: how long the function runs, how much memory is allocated, and how much work has to happen before the first request can be served. AWS Lambda charges per request and per GB-second, so reducing execution time matters, but the configured memory size matters just as much. This is why peak memory is not only an operational metric, but also a cost signal. A function that peaks around 21MB can safely fit inside the 128MB minimum allocation. A function that peaks near 201MB would normally need at least a 256MB allocation to leave a safe margin.

The cost impact becomes clearer if we turn the benchmark into a concrete traffic scenario. Assume 100 million invocations per month on x86 Lambda, excluding the free tier and excluding surrounding services such as API Gateway, CloudWatch Logs, storage, and networking. The request charge is the same for all implementations, so the interesting part is the duration cost. For the estimate below, the scikit-learn Docker version is modeled at 256MB because of its observed memory usage, while compiled Python and compiled Go are modeled at 128MB. Average billed duration is approximated as:

```text
warm p50 latency + cold start rate × cold init time
```

| Cold start rate | scikit-learn Docker | Compiled Python | Compiled Go  | Go vs scikit-learn |
| --------------- | ------------------- | --------------- | ------------ | ------------------ |
| 0%              | $29.33/month        | $20.38/month    | $20.23/month | 31% lower          |
| 1%              | $37.88/month        | $21.31/month    | $20.39/month | 46% lower          |
| 10%             | $114.75/month       | $29.77/month    | $21.86/month | 81% lower          |

This table also shows why a latency chart alone does not fully explain the deployment impact. In the warm-dominated case, request charges become the floor, so the total bill cannot fall by 12x even though median latency does. The compute component does fall almost that much: in the 1% cold-start scenario, moving from scikit-learn Docker to compiled Go reduces duration cost from $17.88/month to $0.39/month, a roughly 98% reduction. Under burstier traffic, the difference is larger because cold initialization becomes part of the cost profile. Reducing cold starts from 2.05 seconds to 78.3 milliseconds changes both user-visible latency and billed initialization work.

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/barufa/barufa.github.io/refs/heads/main/assets/img/treecompiler_benchmark.png" alt="Benchmark" style="max-width: 100%; height: auto;" />
</div>

The broader point is not that every tree model will save exactly this amount. The point is that deployment shape changes the cost equation. Removing the training stack from the inference path reduces package size, initialization work, runtime memory pressure, and execution overhead. For low-volume workloads, the dollar difference may be small because the request charge and free tier dominate. For high-volume or bursty workloads, the same structural changes can translate into meaningful reductions in billed compute and a lower need to overprovision memory.

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
