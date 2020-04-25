Chapter 1. The Golem of Prague
================

The first two sections are spent describing some of the general probmes
that statisticians and researchers face is designing statistical tests
and models.

## 1.3 Tools for golem engineering

  - use models for several distinct purpose:
      - designing inquiry
      - extracting information from data
      - making predictions
  - this book focuses on the following tools towards these purposes:
      - Bayesian data analysis
      - model comparison
      - multilevel models
      - graphical causal models
  - this book focuses mostly on code - how to do things (“golem
    engineering”)

### 1.3.1 Bayesian data analysis

  - Bayesian data analysis takes a question in the form of a model and
    uses logic to produce an answer int he form of probability
    distributions.
  - it is like counting the number of ways the data could happen
    according to some assumptions
      - things that can happen more ways are more plausible

### 1.3.2 Model comparison and prediction

  - there are many ways to compare models
  - we will learn about “cross-validation” and “information criteria” as
    metrics of predictive power of a model
  - this will introduce the phenomenon of more complex models making
    worse predictions: “over-fitting”

### 1.3.3. Multilevel models

  - models contain parameters which can sometimes stand-in for other
    missing models
      - given smoe model of how the parameter gets its value, the new
        model can be inserted in place of the parameter
      - this cretes a final model with multiple levels of uncertainty
  - these models are also called “hierarchical,” “random effects,”
    “varying effects,” or “mixed effects” models
  - multilevel models can help fight overfitting using “partial pooling”
    (covered in Chapter 13)
  - they generally apply when there are clusters or groups of
    measureents that may differ from one another

### Graphical causal models

  - one form of prediction, mentioned above, is what will the outcome be
    in the future
  - another type is causal prediction: what process causes the other
      - this is essential knowledge for using a model to intervene in
        the world
