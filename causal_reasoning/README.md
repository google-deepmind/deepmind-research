# 🧠 Algorithms for Causal Reasoning in Probability Trees

*Developed by the AGI Safety Analysis Team @ DeepMind*

---

## 📌 Overview

This repository provides implementations of algorithms for **causal reasoning** using **probability trees** — a powerful yet underexplored framework for modeling causal systems.

### 🔍 What Are Probability Trees?

Probability trees represent **sequential decision processes** or **generative models**, where each branch encodes possible outcomes and their associated probabilities. Unlike causal Bayesian networks, probability trees:

- Are strictly **more expressive**
- Handle **propositional logic**, **interventions**, and **counterfactuals** in a unified way
- Can represent **asymmetric** or **non-graphical** dependencies

---

## 🎯 Project Goals

This work extends the reach of causal reasoning in AI and machine learning to:
- General discrete stochastic processes
- Cases that lie beyond the representational capacity of causal Bayesian networks

It introduces algorithms to evaluate all three levels of **Judea Pearl’s Causal Hierarchy**:

| Level | Name            | Description                                      |
|-------|-----------------|--------------------------------------------------|
| 1️⃣    | Association     | Observing statistical relationships              |
| 2️⃣    | Intervention    | Modeling the effects of external actions         |
| 3️⃣    | Counterfactuals | Asking “what would’ve happened if...” questions |

---

## 📚 Paper

All algorithms and theoretical concepts used in this repository are explained in the research paper:

🔗 **[Algorithms for Causal Reasoning in Probability Trees (arXiv)](https://arxiv.org/abs/2010.12237)**

If you're new to the topic, we recommend reading the **introduction and background sections** of the paper before diving into the code.

---

## 🧪 Interactive Demo (Google Colab)

The core implementation and examples are provided in a **Jupyter notebook**, which you can run in your browser with no setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/causal_reasoning/Causal_Reasoning_in_Probability_Trees.ipynb)

---

## 🗂️ Folder Structure

```plaintext
causal_reasoning/
├── Causal_Reasoning_in_Probability_Trees.ipynb  # Main demo and implementation notebook
└── README.md                                    # Documentation and contribution guidelines
