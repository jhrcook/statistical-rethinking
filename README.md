
# Notes on *Statistical Rethinking: A Bayesian Course with Examples in R and Stan*

(2nd Edition)

The first two chapters of the book are available as a
[PDF](statisticalrethinking2_chapters1and2.pdf) for free.

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

## Notes

[Chapter 1. The Golem of Prague](ch1_the-golem-of-prague.md)
