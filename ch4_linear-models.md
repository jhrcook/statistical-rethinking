Chapter 4. Linear Models
================

  - this chapter introduce linear regresssion as a Bayesian procedure
      - under a probability interpretation (necessary for Bayesian
        word), linear reg. uses a Gaussian distribution for uncertainty
        about the measurement of interest

## 4.1 Why normal distributions are normal

  - example:
      - you have 1,000 people stand at the half-way line of a soccer
        field
      - they each flip a coin 16 times, moving left if the coin comes up
        heads, and right if it comes up tails
      - the final distribution of people around the half-way line will
        be Gaussian even though the underlying model is binomial

### 4.1.1 Normal by addition

  - we can simulate the above example
      - to show that the underlying coin-flip is nothing special, we
        will instead use a random values between -1 and 1 for each
        person to step

<!-- end list -->

``` r
pos <- replicate(1e3, sum(runif(16, -1, 1)))
plot(density(pos))
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` r
set.seed(0)

n_steps <- 16

position_data <- tibble(person = 1:1e3) %>%
    mutate(position = purrr::map(person, ~ c(0, cumsum(runif(n_steps, -1, 1))))) %>%
    unnest(position) %>%
    group_by(person) %>%
    mutate(step = 0:n_steps) %>%
    ungroup()

walks_plot <- position_data %>%
    ggplot(aes(x = step, y = position, group = person)) +
    geom_line(alpha = 0.2, size = 0.1, color = "dodgerblue") +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "step", y = "position", 
         title = "Random-walks")

step_densities <- position_data %>%
    filter(step %in% c(4, 8, 16)) %>%
    ggplot(aes(x = position)) +
    facet_wrap(~ step, scales = "free_y", nrow = 1) +
    geom_density(fill = "grey90", color = "grey40") +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
    labs(x = "position",
         y = "density",
         title = "Position distribution at several steps")

walks_plot / step_densities + plot_layout(heights = c(3, 2))
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

### 4.1.2 Normal by multiplication

  - another example of how to get a normal distribution:
      - the growth rate of an organism is influenced by a dozen loci,
        each with several alleles that code for more growth
      - suppose that these loci interact with one another, and each
        increase growth by a percentage
      - therefore, their effects multiply instead of add
      - below is a simulation of sampling growth rates
      - this distribution is approximately normal because the
        multiplication of small numbers is approximately the same as
        addition

<!-- end list -->

``` r
# A single growth rate
prod(1 + runif(12, 0, 0.1))
```

    ## [1] 1.846713

``` r
growth <- replicate(1e4, prod(1 + runif(12, 0, 0.1)))
dens(growth, norm.comp = TRUE)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

### 4.1.3 Normal by log-multiplication

  - large deviates multiplied together produce Gaussian distributions on
    the log-scale

<!-- end list -->

``` r
growth <- replicate(1e4, prod(1 + runif(12, 0, 0.5)))
dens(growth, norm.comp = TRUE, main = "Not scaled")
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
dens(log(growth), norm.comp = TRUE, main = "Log-scaled")
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

### 4.1.4 Using Gaussian distributions

  - we will build models of measurements as aggregations of normal
    distributions
  - this is appropriate because:
      - the world is full of approximately normal distributions
      - we often are quite ignorant of the underlying distribution so
        modeling it as a mean and variance is often the best we can do

## 4.2 A language for describing models

  - here is an outline of the process commonly used:
    1.  recognize a set of measurements to predict or understand - the
        *outcome* variables
    2.  for each variable, define a likelihood distribution that defines
        the plausibility of individual observations
          - this is always Gaussian for linear regression
    3.  recognize a set of other measurements to use to predict or
        understand the outcome - the *predictory* variables
    4.  relate the shape of the likelihood distribtuion to the predictor
        variables
    5.  choose priors for all parameters in the model; this is the
        initial state of the model before seeing any data
    6.  summarise the model with math expressions; for example:

\[
\text{outcome}_i \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \beta \times \text{predictor}_i \\
\beta \sim \text{Normal}(0, 10) \\
\sigma \sim \text{HalfCauchy}(0, 1)
\]

### 4.2.1 Re-describing the globe tossing model

  - this example was trying to estimate the proportion of water \(p\) of
    a globe by tossing it and counting of how often our finger was on
    water upon catching the globe
      - it could be described as such where \(w\) is the number of
        waters observed, \(n\) is the total number of tosses, and \(p\)
        is the proportion of the water on the globe

\[
w \sim \text{Binomial(n,p)} \\
p \sim \text{Uniform}(0,1)
\]

  - this should be read as:
      - “The count \(w\) is distributed binomially with sample size
        \(n\) and probability \(p\).”
      - “The prior for \(p\) is assumed to be uniform between zero and
        one.”

## 4.3 A Gaussian model of height

  - we will now build a linear regression model
      - this section will build the scaffold
      - the next will construct the predictor variable
  - model a single measurement variable as a Gaussian distribution
      - two parameters: \(\mu\) = mean; \(\sigma\) = standard deviation
      - Bayesian updating will consider each possible combination of
        \(\mu\) and \(\sigma\) and provide a socre for the plausibility
        of each

### 4.3.1 The data

  - we will use the `Howll1` data from the ‘rethinking’ package
      - we will use height information for peple older that 18 years

<!-- end list -->

``` r
data("Howell1")
d <- Howell1
str(d)
```

    ## 'data.frame':    544 obs. of  4 variables:
    ##  $ height: num  152 140 137 157 145 ...
    ##  $ weight: num  47.8 36.5 31.9 53 41.3 ...
    ##  $ age   : num  63 63 65 41 51 35 32 27 19 54 ...
    ##  $ male  : int  1 0 0 1 0 1 0 1 0 1 ...

``` r
d2 <- d[d$age >= 18, ]
nrow(d2)
```

    ## [1] 352

### 4.3.2 The model

  - the goal is to model these values using a Gaussian distribution

<!-- end list -->

``` r
dens(d2$height)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

  - our model is

\[
h_i \sim \text{Normal}(\mu, \sigma) \quad \text{or} \quad h_i \sim \mathcal{N}(\mu, \sigma)
\]

  - the priors for the model parameters are below
      - the mean and s.d. for the normal distribution for \(\mu\) were
        just chosen by the author as likely a good guess for the average
        heights

\[
\mu \sim \mathcal{N}(178, 20) \\
\sigma \sim \text{Uniform}(0, 50)
\]

  - it is often good to plot the priors

<!-- end list -->

``` r
curve(dnorm(x, 178, 20), from = 100, to = 250)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
curve(dunif(x, 0, 50), from = 10, to = 60)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

  - we can sample from the priors to build our “expected” distribution
    of heights
      - its the relative plausibility of different heights before seeing
        any data

<!-- end list -->

``` r
sample_mu <- rnorm(1e4, 178, 20)
sample_sigma <- runif(1e4, 0, 50)

prior_h <- rnorm(1e4, sample_mu, sample_sigma)

dens(prior_h)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

### 4.3.3 Grid approximation of the posteior distribution

  - as an example, we will map the posterior distribution using brute
    force
      - later, we will switch to the quadratic approximation that we
        will use for the next few chapters
      - we use a few shortcuts here, including summing the log
        likelihood instead of multiplying the likelihoods

<!-- end list -->

``` r
mu_list <- seq(from = 140, to = 160, length.out = 200)
sigma_list <- seq(from = 4, to = 9, length.out = 200)

post <- expand.grid(mu = mu_list, sigma = sigma_list)
head(post)
```

    ##         mu sigma
    ## 1 140.0000     4
    ## 2 140.1005     4
    ## 3 140.2010     4
    ## 4 140.3015     4
    ## 5 140.4020     4
    ## 6 140.5025     4

``` r
set.seed(0)

post$LL <- pmap_dbl(post, function(mu, sigma, ...) {
    sum(dnorm(
        d2$height,
        mean = mu,
        sd = sigma,
        log = TRUE
    ))
})

post$prod <- post$LL + dnorm(post$mu, 178, 20, log = TRUE) + dunif(post$sigma, 0, 500, log = TRUE)

post$prob <- exp(post$prod - max(post$prod))
```

``` r
post %>% 
    as_tibble() %>%
    ggplot(aes(x = mu, y = sigma, z = prob)) +
    geom_contour()
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
as_tibble(post) %>%
    ggplot(aes(x = mu, y = sigma, fill = prob)) +
    geom_tile(color = NA) +
    scale_fill_viridis_b()
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

### 4.3.4 Sampling from the posterior

  - we sample from the posterior just like normal, except now we get
    pairs of parameters

<!-- end list -->

``` r
sample_rows <- sample(1:nrow(post), 
                      size = 1e4, 
                      replace = TRUE, 
                      prob = post$prob)
sample_mu <- post$mu[sample_rows]
sample_sigma <- post$sigma[sample_rows]
tibble(mu = sample_mu, sigma = sample_sigma) %>%
    ggplot(aes(x = mu, y = sigma)) +
    geom_jitter(size = 1, alpha = 0.2, color = "grey30", 
                width = 0.1, height = 0.1)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

  - now we can describe these parameters just like data
      - the distributions are the results

<!-- end list -->

``` r
tibble(name = c(rep("mu", 1e4), rep("sigma", 1e4)),
       value = c(sample_mu, sample_sigma)) %>%
    ggplot(aes(x = value)) +
    facet_wrap(~ name, nrow = 1, scales = "free") +
    geom_density(fill = "grey90", alpha = 0.5)
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
cat("HPDI of mu:\n")
```

    ## HPDI of mu:

``` r
HPDI(sample_mu)
```

    ##    |0.89    0.89| 
    ## 153.8693 155.1759

``` r
cat("\nHPDI of sigma:\n")
```

    ## 
    ## HPDI of sigma:

``` r
HPDI(sample_sigma)
```

    ##    |0.89    0.89| 
    ## 7.291457 8.221106

4.3.5 Fitting the model with `map()`

**Note that the `map()` function has been changed to `quap()` in the 2nd
Edition of the course.**

  - now we can use the `quap()` function to conduct the quadratic
    approximation of the posterior
  - recall that this is the model definition:

\[
h_i \sim \text{Normal}(\mu, \sigma) \\
\mu \sim \text{Normal}(178, 20) \\
\sigma \sim \text{Uniform}(0, 50)
\]

  - first, we copy this formula into an `alist`

<!-- end list -->

``` r
formula_list <- alist(
    height ~ dnorm(mu, sigma),
    mu ~ dnorm(178, 20),
    sigma ~ dunif(0, 50)
)
formula_list
```

    ## [[1]]
    ## height ~ dnorm(mu, sigma)
    ## 
    ## [[2]]
    ## mu ~ dnorm(178, 20)
    ## 
    ## [[3]]
    ## sigma ~ dunif(0, 50)

  - then we can fit the model to the data using `quap()` and the data in
    `d2`

<!-- end list -->

``` r
m4_1 <- quap(formula_list, data = d2)
summary(m4_1)
```

    ##             mean        sd       5.5%     94.5%
    ## mu    154.607032 0.4120116 153.948558 155.26551
    ## sigma   7.731651 0.2914160   7.265912   8.19739

### 4.3.6 Sampling from a map fit

  - the quadratic approximation to a posterior dist. with multiple
    parameters is just a multidimensional Gaussian distribution
      - therefore, it can be described by its variance-covariance matrix

<!-- end list -->

``` r
vcov(m4_1)
```

    ##                 mu        sigma
    ## mu    0.1697535649 0.0002182476
    ## sigma 0.0002182476 0.0849232811

  - the variance-covariance matrix tells us how the parameters relate to
    each other
  - it can be decomposed into two pieces:
    1.  the vector of varainces for the parameters
    2.  a correlation matrix that tells us how the changes in one
        parameter lead to a correlated change in the others

<!-- end list -->

``` r
cat("Covariances:\n")
```

    ## Covariances:

``` r
diag(vcov(m4_1))
```

    ##         mu      sigma 
    ## 0.16975356 0.08492328

``` r
cat("\nCorrelations:\n")
```

    ## 
    ## Correlations:

``` r
cov2cor(vcov(m4_1))
```

    ##                mu       sigma
    ## mu    1.000000000 0.001817718
    ## sigma 0.001817718 1.000000000

  - instead of sampling single values from a simple Gaussian
    distribution, we sample vectors of values from a multi-dimensional
    Gaussian distribution
      - the `extract.samples()` function from ‘rethinking’ does this for
        us

<!-- end list -->

``` r
post <- extract.samples(m4_1, n = 1e4)
head(post)
```

    ##         mu    sigma
    ## 1 154.1145 7.599922
    ## 2 154.8011 7.305567
    ## 3 154.7124 7.813402
    ## 4 154.6557 7.511392
    ## 5 154.9872 8.218177
    ## 6 154.5500 7.601779

``` r
precis(post)
```

    ##             mean        sd       5.5%      94.5%    histogram
    ## mu    154.605370 0.4174050 153.935029 155.272336      ▁▁▅▇▂▁▁
    ## sigma   7.733771 0.2884477   7.275115   8.193431 ▁▁▁▂▅▇▇▃▁▁▁▁

## 4.4 Adding a predictor

  - above, we created a Gaussian model of height in a population of
    adults
      - by adding a predictor, we can make a linear regression
      - for this example, we will see how height covaries with height

<!-- end list -->

``` r
d2 %>%
    ggplot(aes(x = weight, y = height)) +
    geom_point()
```

![](ch4_linear-models_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

### 4.4.1 The linear model strategy

  - the strategy is to make the parameter for the mean of a Gaussian
    dist., \(\mu\), into a linear function of the predictor variable and
    other, new parameters we make
  - some of the parameters of the linear model indicate the strength of
    association between the mean of the outcome and the value of the
    predictor
      - the posterior provides relative plausibilities of the different
        possible strengths of association
  - here is the formula for the linear model
      - let \(x\) be the mathematical name for the weight measurements
      - we model the mean \(\mu\) as a function of \(x\)

\[
h_i \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \beta x_i \\
\alpha \sim \text{Normal}(178, 100) \\
\beta \sim \text{Normal}(0, 10) \\
\sigma \sim \text{Uniform}(0, 50) \\
\]

### 4.4.2 Fitting the model

``` r
m4_3 <- quap(
    alist(
        height ~ dnorm(mu, sigma),
        mu <- a + b*weight,
        a ~ dnorm(178, 100),
        b ~ dnorm(0, 10),
        sigma ~ dunif(0, 50)
    ),
    data = d2
)
summary(m4_3)
```

    ##              mean         sd        5.5%       94.5%
    ## a     113.9033852 1.90526701 110.8584005 116.9483699
    ## b       0.9045063 0.04192009   0.8375099   0.9715027
    ## sigma   5.0718671 0.19115324   4.7663673   5.3773669

4.4.3 Interpreting the model fit
