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
