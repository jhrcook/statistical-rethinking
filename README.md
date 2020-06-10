
# Notes on *Statistical Rethinking: A Bayesian Course with Examples in R and Stan*

The first two chapters of the 2nd edition of this book are available as
a [PDF](statisticalrethinking2_chapters1and2.pdf) for free. The rest of
the book I used the first edition as it is available through my schools
library.

## Introduction

This book is an introduction to Bayesian statistics, focussing on
providing an applicable education in R. This reporisoty contains my
notes; these notes are not a throughough recapitulation of the book, but
instead acting as a combination of a reference and a playground for
myself.

## Setting-up

This course includes an R packages called ‘rethinking’\](). It can be
installed as follows.

``` r
# Dependencies
install.packages(c("coda", "mvtnorm", "devtools", "loo", "dagitty"))

# Course package
devtools::install_github("rmcelreath/rethinking")
```

## Chapter Notes

[Chapter 1. The Golem of Prague](ch1_the-golem-of-prague.md)

[Chapter 2. Small Worlds and Large
Worlds](ch2_small-worlds-and-large-worlds.md)

[Chapter 3. Sampling the Imaginary](ch3_sampling-the-imaginary.md)

[Chapter 4. Linear Models](ch4_linear-models.md)

[Chapter 5. Multivariate Linear
Models](ch5_multivariate-linear-models.md)

[Chapter 6. Overfitting, Regularization, and Information
Criteria](ch6_overfitting-regularization-and-information-criteria.md)

[Chapter 7. Interactions](ch7_interactions.md)

[Chapter 8. Markov Chain Monte Carlo](ch8_markov-chain-monte-carlo.md)

[Chapter 9. Big Entropy and the Generalized Linear
Model](ch9_big-entropy-and-the-generalized-linear-model.md)

[Chapter 10. Counting and
Classification](ch10_counting-and-classification.md)

[Chapter 11. Monsters and Mixtures](ch11_monsters-and-mixtures.md)

[Chapter 12. Multilevel Models](ch12_multilevel-models.md)

[Chapter 13. Adventures in Covariance](ch13_adventures-in-covariance.md)

-----

## Other R packages for Bayes

Below are some of the common packages for using Baysian statistics in R.
The purpose of this collection is to serve as a reference for future
work. As I work through their vignettes (and possible other associated
tutorals/articles), the markdown files will be linked below.

**[‘rstanarm’](https://mc-stan.org/rstanarm/index.html)**

**[‘brms’](https://paul-buerkner.github.io/brms/)**

**[‘ggmcmc’](https://cran.r-project.org/web/packages/ggmcmc/index.html)**

**[‘tidybayes’](http://mjskay.github.io/tidybayes)**

**[‘bayetestR’](https://easystats.github.io/bayestestR/)** (part of the
‘easystats’ suite of packages)
