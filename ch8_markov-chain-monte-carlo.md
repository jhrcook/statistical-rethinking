Chapter 8. Markov Chain Monte Carlo
================

  - estimation of posterior probability distributions using a stochastic
    process called \*Markov chain Monte Carlo (MCMC)\*\* estimation
      - sample directly from the posterior instead of approximating
        curves (like the quadratic approximation)
      - allows for models that do not assume a multivariate normality
          - can use generalized linear and multilevel models
  - use \*\*Stan\* to fit these models

## 8.1 Good King Markov and His island kingdom

  - a tale of the Good King Markov
      - he was kings of a ring of 10 islands
      - the second island had twice the population of the first, the
        third had three times the populaiton of the first, etc.
      - the king wanted to visit the islands in proportion to the
        population size, but didn’t want to have to keep track of a
        schedule
      - he would also only travel between adjacent islands
      - he used the *Metropolis algorithm* to decide which island to
        visit next
        1.  the king decides to stay or travel by a cion flip
        2.  if the coin is heads, the king considers moving clockwise;
            if tails, he considers moving counter-clockwise; call this
            next island the “proposal” island
        3.  to decide whether not to move, the king collects the number
            of seashells in proportion to the population size of the
            proposed island, and collects the number of stones relative
            to the population of the current island
        4.  if there are more seashells, the king moves to the poposed
            island; else he discards the number of stones equal to the
            number of seashells and randomly selects from the remaining
            seashells and stones; if he selects a shell, he travels to
            the proposed island, else he stays
  - below is a simulation of this process

<!-- end list -->

``` r
set.seed(0)

current_island <- 10
num_weeks <- 1e5
visited_islands <- rep(0, num_weeks)

flip_coin_to_decide_proposal_island <- function(x) {
    a <- sample(c(0, 1), 1)
    if (a == 0) {
        y <- x - 1
    } else {
        y <- x + 1
    }
    
    if (y == 0) {
        y <- 10
    } else if (y == 11) {
        y <- 1
    }
    return(y)
}

for (wk in seq(1, num_weeks)) {
    visited_islands[[wk]] <- current_island
    
    proposal_island <- flip_coin_to_decide_proposal_island(current_island)
    if (proposal_island > current_island) {
        current_island <- proposal_island
    } else {
        shells <- rep("shell", proposal_island)
        stones <- rep("stone", current_island - proposal_island)
        selection <- sample(unlist(c(shells, stones)), 1)
        if (selection == "shell") {
            current_island <- proposal_island
        }
    }
}
```

``` r
tibble(wk = seq(1, num_weeks),
       islands = visited_islands) %>%
    ggplot(aes(x = factor(islands))) + 
    geom_bar(fill = "skyblue4") +
    labs(x = "islands",
         y = "number of weeks visited",
         title = "Distribution of King Markov's visits")
```

![](ch8_markov-chain-monte-carlo_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
tibble(wk = seq(1, num_weeks),
       islands = visited_islands) %>%
    slice(1:400) %>%
    ggplot(aes(x = wk, y = islands)) +
    geom_line(color = "skyblue4", alpha = 0.5) +
    geom_point(color = "skyblue4", size = 0.7) +
    scale_y_continuous(breaks = c(1:10)) +
    labs(x = "week number", y = "island",
         title = "The path of King Markov",
         subtitle = "Showing only the first 400 weeks")
```

![](ch8_markov-chain-monte-carlo_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

  - this algorithm still works if the king is equally likely to propose
    a move to any island from the current island
      - still use the proportion of the islands’ populations as the
        probability of moving
  - at any point, the king only needs to know the population of the
    current island and the proposal island

## 8.2 Markov chain Monte Carlo

  - the *Metropolis algorithm* used above is an example of a *Markov
    chain Monte Carlo*
      - the goal is to draw samples from an unknown and complex target
        distribution
          - “islands”: the parameter values (they can be continuous,
            too)
          - “population sizes”: the posterior probabilities at each
            parameter value
          - “week”: samples taken from the joint posterior of the
            parameters
      - we can use the samples from the Metropolis algorithm just like
        any other sampled distributions so far
  - we will also cover *Gibbs sampling* and *Hamiltonian Monte Carlo*

### 8.2.1 Gibbs sampling

  - can achieve a more efficient sampling procedure
      - *adaptive proposals*: the distribution of proposed parameter
        values adjusts itself intelligently depending upon the parameter
        values at the moment
      - the adaptive proposals depend on using particular combinations
        of prior distributions and likelihoods known as conjugate pairs
          - these pairs have analytical solutions for the posterior of
            an individual parameter
          - can use these solutions to make smart jumps around the joint
            posterior
  - Gibbs sampling is used in BUGS (Bayesian inference Using Gibbs
    Sampling) and JAGS (Just Another Gibbs Sampler)
  - limitations:
      - don’t always want to use the conjugate priors
      - Gibbs sampling becomes very innefficient for models with
        hundreds or thousands of parameters

### 8.2.2 Hamiltonian Monte Carlo (HMC)

  - more computationally expensive than Gibbs sampling and the
    Metropolis algorithm, but more efficient
      - doesn’t require as many samples to describe the posterior
        distribution
  - an analogy using King Monty:
      - king of a continuous stretch of land, not discrete islands
      - wants to visit his citizens in proportion to their local density
      - decides to travel back and forth across the length of the
        country, slowing down the vehicle when houses grow more dense,
        and speeing up when the houses are more sparse
      - requires knowledge of how quickly the population is changing at
        the current location
  - in this analogy:
      - “royal vehicle”: the current vector of parameter values
      - the log-posterior forms a bowl with the MAP at the nadir
      - the vehicle sweeps across the bowl, adjust the speed in
        proportion to the height on the bowl
  - HMC does run a sort of physics simultation
      - a vector of parameters gives the position of a frictionless
        particle
      - the log-posterior provides the surface
  - limitations:
      - HMC requires continuous parameters
      - HMC needs to be tuned to a particular model and its data
          - the particle needs mass (so it can have momentum)
          - other hyperparameters of MCMC is handeled by Stan

## 8.3 Easy HMC: `map2stan`

  - `map2stan()` provides an interface to Stan similar to how we have
    used `quap()`
      - need to preprocess and variables transformations
      - input data frame can only have columns used in the formulae
  - example: look at terrain ruggedness data

<!-- end list -->

``` r
data("rugged")
dd <- as_tibble(rugged) %>%
    mutate(log_gdp = log(rgdppc_2000)) %>%
    filter(!is.na(rgdppc_2000))
```

  - fit a model to predict log-GDP with terrain ruggedness, continent,
    and the interaction of the two
  - first fit with `quap()` like before

<!-- end list -->

``` r
m8_1 <- quap(
    alist(
        log_gdp ~ dnorm(mu, sigma),
        mu <- a + bR*rugged + bA*cont_africa + bAR*rugged*cont_africa,
        a ~ dnorm(0, 100),
        bR ~ dnorm(0, 10),
        bA ~ dnorm(0, 10),
        bAR ~ dnorm(0, 10),
        sigma ~ dunif(0, 10)
    ),
    data = dd
)
precis(m8_1)
```

    ##             mean         sd       5.5%       94.5%
    ## a      9.2227717 0.13798197  9.0022499  9.44329359
    ## bR    -0.2026506 0.07646932 -0.3248634 -0.08043786
    ## bA    -1.9469424 0.22450135 -2.3057389 -1.58814589
    ## bAR    0.3929006 0.13004832  0.1850583  0.60074296
    ## sigma  0.9326829 0.05058184  0.8518433  1.01352241

### 8.3.1 Preparation

  - must:
      - do any transformations beforehand (e.g. logarithm, squared,
        etc.)
      - reduce data frame to only used variables

<!-- end list -->

``` r
dd_trim <- dd %>%
    select(log_gdp, rugged, cont_africa)
```

### 8.3.2 Estimation

  - can now fit the model with HMC using `map2stan()`

<!-- end list -->

``` r
m8_1stan <- map2stan(
    alist(
        log_gdp ~ dnorm(mu, sigma),
        mu <- a + bR*rugged + bA*cont_africa + bAR*rugged*cont_africa,
        a ~ dnorm(0, 100),
        bR ~ dnorm(0, 10),
        bA ~ dnorm(0, 10),
        bAR ~ dnorm(0, 10),
        sigma ~ dcauchy(0, 2)
    ),
    data = dd_trim
)
```

    ## Trying to compile a simple C file

    ## Running /Library/Frameworks/R.framework/Resources/bin/R CMD SHLIB foo.c
    ## clang -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG   -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/Rcpp/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/unsupported"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/BH/include" -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/src/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/"  -I"/Library/Frameworks/R.framework/Versions/3.6/Resources/library/rstan/include" -DEIGEN_NO_DEBUG  -D_REENTRANT  -DBOOST_DISABLE_ASSERTS -DBOOST_PENDING_INTEGER_LOG2_HPP -include stan/math/prim/mat/fun/Eigen.hpp   -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -I/usr/local/include  -fPIC  -Wall -g -O2  -c foo.c -o foo.o
    ## In file included from <built-in>:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:13:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Dense:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Core:88:
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:613:1: error: unknown type name 'namespace'
    ## namespace Eigen {
    ## ^
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/src/Core/util/Macros.h:613:16: error: expected ';' after top level declarator
    ## namespace Eigen {
    ##                ^
    ##                ;
    ## In file included from <built-in>:1:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/StanHeaders/include/stan/math/prim/mat/fun/Eigen.hpp:13:
    ## In file included from /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Dense:1:
    ## /Library/Frameworks/R.framework/Versions/3.6/Resources/library/RcppEigen/include/Eigen/Core:96:10: fatal error: 'complex' file not found
    ## #include <complex>
    ##          ^~~~~~~~~
    ## 3 errors generated.
    ## make: *** [foo.o] Error 1
    ## 
    ## SAMPLING FOR MODEL 'f6693cddea054695f17c31793a066d41' NOW (CHAIN 1).
    ## Chain 1: 
    ## Chain 1: Gradient evaluation took 6.2e-05 seconds
    ## Chain 1: 1000 transitions using 10 leapfrog steps per transition would take 0.62 seconds.
    ## Chain 1: Adjust your expectations accordingly!
    ## Chain 1: 
    ## Chain 1: 
    ## Chain 1: Iteration:    1 / 2000 [  0%]  (Warmup)
    ## Chain 1: Iteration:  200 / 2000 [ 10%]  (Warmup)
    ## Chain 1: Iteration:  400 / 2000 [ 20%]  (Warmup)
    ## Chain 1: Iteration:  600 / 2000 [ 30%]  (Warmup)
    ## Chain 1: Iteration:  800 / 2000 [ 40%]  (Warmup)
    ## Chain 1: Iteration: 1000 / 2000 [ 50%]  (Warmup)
    ## Chain 1: Iteration: 1001 / 2000 [ 50%]  (Sampling)
    ## Chain 1: Iteration: 1200 / 2000 [ 60%]  (Sampling)
    ## Chain 1: Iteration: 1400 / 2000 [ 70%]  (Sampling)
    ## Chain 1: Iteration: 1600 / 2000 [ 80%]  (Sampling)
    ## Chain 1: Iteration: 1800 / 2000 [ 90%]  (Sampling)
    ## Chain 1: Iteration: 2000 / 2000 [100%]  (Sampling)
    ## Chain 1: 
    ## Chain 1:  Elapsed Time: 0.272953 seconds (Warm-up)
    ## Chain 1:                0.275488 seconds (Sampling)
    ## Chain 1:                0.548441 seconds (Total)
    ## Chain 1:

    ## Computing WAIC

``` r
precis(m8_1stan)
```

    ##             mean         sd       5.5%       94.5%    n_eff     Rhat4
    ## a      9.2182611 0.14034806  8.9999067  9.43564460 381.7565 1.0081349
    ## bR    -0.2002538 0.08066235 -0.3253604 -0.07468424 348.7989 1.0179494
    ## bA    -1.9413043 0.23158324 -2.2990252 -1.56245440 475.1342 0.9990503
    ## bAR    0.3897727 0.14112017  0.1729353  0.60830114 452.2939 0.9999791
    ## sigma  0.9504754 0.05263645  0.8721904  1.03751376 454.5194 1.0041046

  - a half-Cauchy prior was used for \(\sigma\)
      - a uniform distribution would work here, too
      - it is a useful, “thick-tailed” probability
      - related to the Student \(t\) distribution
      - can think of it as a weakly-regularizing prior for the standard
        deviation
  - the estimates from this model are simillar to those from the
    quadratic prior
  - a few differences to note about the output of `precis()`
    (`summary()`):
      - the *highest probability density intervals* (HPDI) are shown,
        not just percentiles intervals (PI), like before
      - two new columns (they will be discussed further later):
          - `n_eff`: a crude estimate of the number of independent
            samples that were collected
          - `Rhat4`: an estimate of the convergence of the Markov chains
            (1 is good)

### 8.3.3 Sampling again, in parallel
