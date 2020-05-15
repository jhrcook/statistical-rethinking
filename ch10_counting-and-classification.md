Chapter 10. Counting and Classification
================

  - information if often thrown away by using ratios of counts instead
    of the counts themselves
      - 10/20 and 1/2 are the same ratio, but the first has more
        information
  - there is friction in using the count data instead of the proportions
      - interpretation is less intuitive
  - this chapter covers the two most popular count distributions:
    1.  *binomial regression*: model a binary classification
          - dead/alive, accept/reject 2 *poisson regression*: models
            outcome without a known maximum
          - a binomial models with a very large maximum and samll
            probability per trial
          - number of elephants in Kenya, number of people who apply to
            graduate school

## 10.1 Binomial regression

  - the following formula where \(y\) is a count, \(p\) is the
    probability of success in a trial, \(n\) is the number of trials \[
    y \sim \text{Binomial}(n, p)
    \]

  - the two most common GLMs that use binomial likelihood functions:
    
    1.  *logistic regression*: data are in single-trial cases and the
        outcome is only 0 or 1
    2.  *aggregated binomial regression*: when individual trials with
        the same covariate values are aggregated together
          - the outcome can take a value of 0 or any positive integer up
            to \(n\) number of trials

  - both GLMs use the same logit link function
    
      - so both are sometimes called logistic regression and they can be
        converted between each other

## 10.1.1 Logisitc regression: Prosocial chimpanzees

  - example experimental data
      - measure the prosocial behaviour of chimps
      - a focal chimp has the option to pull two levers where the first
        only gives the focal chimp food and the other gives the focal
        chimp food, but not the other chimp
      - therefore, the focal chimp always gets food and decides if the
        other chimp gets food
      - the control condition is to not have another chimp and the
        partner condition is to have the second chimp
      - the two choices are swapped from the left and right hand sides
        to detect any handedness of the focal chimps

<!-- end list -->

``` r
data("chimpanzees")
d <- as_tibble(chimpanzees) %>% janitor::clean_names()
d
```

    ## # A tibble: 504 x 8
    ##    actor recipient condition block trial prosoc_left chose_prosoc pulled_left
    ##    <int>     <int>     <int> <int> <int>       <int>        <int>       <int>
    ##  1     1        NA         0     1     2           0            1           0
    ##  2     1        NA         0     1     4           0            0           1
    ##  3     1        NA         0     1     6           1            0           0
    ##  4     1        NA         0     1     8           0            1           0
    ##  5     1        NA         0     1    10           1            1           1
    ##  6     1        NA         0     1    12           1            1           1
    ##  7     1        NA         0     2    14           1            0           0
    ##  8     1        NA         0     2    16           1            0           0
    ##  9     1        NA         0     2    18           0            1           0
    ## 10     1        NA         0     2    20           0            1           0
    ## # … with 494 more rows

  - we will focus on the columns:
      - `pulled_left`: the outcome to predict
      - `prosoc_left`: a predictor for if the left-hand lever was the
        prosocial option
      - `condition`: contains a 1 for when there was a second partner
        chimp
  - the model we will fit:
      - \(L\): indicates if the left-hand lever was pulled
      - \(P\): indicates if the left-hand option was pro-social
      - \(C\): indicates whether or not the condition was with the
        partner

\[
L_i \sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) = \alpha + (\beta_P + \beta_{PC} C_i)P_i \\
\alpha \sim \text{Normal}(0, 10) \\
\beta_P \sim \text{Normal}(0, 10) \\
\beta_PC \sim \text{Normal}(0, 10)
\]

  - this model includes an interaction term for the left-hand option
    being pro-social and whether or not there is a second chimp
      - also there is no main effect of the `condition` \(C_i\) because
        we do not expect the precense of a second chimp on its own to
        predict whether the focal chimp pulls the left lever
  - the priors are gently regularizing
  - as comparitive measures of overfitting, fit two other models with
    fewer parameters
      - one with just an intercept
      - one without only th `prosoc_left` predictor (no predictor for
        `condition` whether there is a second chimp)

\[
L_i \sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) = \alpha \\
\alpha \sim \text{Normal}(0, 10)
\]

\[
L_i \sim \text{Binomial}(1, p_i) \\
\text{logit}(p_i) = \alpha + \beta_P P_i \\
\alpha \sim \text{Normal}(0, 10) \\
\beta_P \sim \text{Normal}(0, 10)
\]

  - first we will inspect the simplest model, the one with only an
    intercept

<!-- end list -->

``` r
m10_1 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a,
        a ~ dnorm(0, 10)
    ),
    data = d
)

precis(m10_1)
```

    ##        mean         sd     5.5%    94.5%
    ## a 0.3201415 0.09022718 0.175941 0.464342

  - \(\alpha\) is on the scale of log-odds
      - to get it to probability scale, must use the inverse link
        function, the logistic
      - the ‘rethinking’ package offers the function `logisitic()` to do
        this, but below I just show the calculation for education
        reasons

<!-- end list -->

``` r
# MAP
1 / (1 + exp(-0.32))
```

    ## [1] 0.5793243

``` r
# 89% interval
c(1 / (1 + exp(-0.18)), 1 / (1 + exp(-0.46)))
```

    ## [1] 0.5448789 0.6130142

``` r
logistic
```

    ## function (x) 
    ## {
    ##     p <- 1/(1 + exp(-x))
    ##     p <- ifelse(x == Inf, 1, p)
    ##     p
    ## }
    ## <bytecode: 0x7facfb897f90>
    ## <environment: namespace:rethinking>

  - \(\text{logistic}(0.32) \approx 0.58\) means that the probability of
    pulling the left-hand lever was 0.58 with an 89% interval of 0.54 to
    0.61
      - the chimps had a tendency to favor the left without any other
        information
  - the following two code chunks fit the other two models proposed
    above

<!-- end list -->

``` r
m10_2 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a + bp*prosoc_left,
        a ~ dnorm(0, 10),
        bp ~ dnorm(0, 10)
    ),
    data = d
)

precis(m10_2)
```

    ##         mean        sd       5.5%     94.5%
    ## a  0.0477090 0.1260040 -0.1536697 0.2490877
    ## bp 0.5573081 0.1823154  0.2659328 0.8486833

``` r
m10_3 <- quap(
    alist(
        pulled_left ~ dbinom(1, p),
        logit(p) <- a + (bp + bpc*condition)*prosoc_left,
        a ~ dnorm(0, 10),
        bp ~ dnorm(0, 10),
        bpc ~ dnorm(0, 10)
    ),
    data = d
)

precis(m10_3)
```

    ##            mean        sd       5.5%     94.5%
    ## a    0.04771766 0.1260040 -0.1536611 0.2490964
    ## bp   0.60967089 0.2261462  0.2482456 0.9710962
    ## bpc -0.10396684 0.2635904 -0.5252352 0.3173015

``` r
compare(m10_1, m10_2, m10_3)
```

    ##           WAIC       SE    dWAIC       dSE     pWAIC    weight
    ## m10_2 680.4648 9.165323 0.000000        NA 1.9807109 0.7106338
    ## m10_3 682.3826 9.403175 1.917804 0.8137132 3.0213904 0.2723955
    ## m10_1 687.9341 7.070955 7.469337 6.0718121 0.9966656 0.0169707

``` r
plot(compare(m10_1, m10_2, m10_3))
```

![](ch10_counting-and-classification_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

  - from the WAIC, we can see that `m10_3` likely overfits a bit because
    its WAIC is greater than `m10_2`
      - though the difference in WAIC is samll, the difference standard
        error `dSE` is very small and suggests it is a real difference
  - but `m10_3` should not just be rejected, it still reflects the
    structure of the experiment
      - we do want to see why `m10_3` performs worse than `m10_2`
  - the estimates for `m10_3` show a negative interaction term with a
    large 89% interval
      - suggests the chimps don’t care too much about the presence of
        another chimp
      - they do prefer to pull the prosocial option, though, because
        that estimate is 0.61 with an 89% interval well above 0
  - to understand the impact of the estimate 0.61 for `bp`, must
    distinguish between the *absolute effect* and the *relative effect*
      - *absolute effect*: the change in the probability of the outcome,
        depending on all of the parameters
          - tells us the practical impact of a change in a predictor
      - *relative effect*: the proportional changes induced by a change
        in the predictor
          - the author claims that this effect can be misleading because
            they ignore the other parameters
