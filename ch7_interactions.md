Chapter 7. Interactions
================

  - so far, we have assumed that each predictor has an independent
    association with the mean of the outcome
      - now we will look at conditioning this estimate on another
        predictor using *interactions*
  - fitting a model with interactions is easy, but understanding them
    can be harder

## 7.1 Building an interaction

  - for examples, we will use information on the economy countries and
    geographic properties

<!-- end list -->

``` r
data("rugged")

d <- rugged
d$log_gdp <- log(d$rgdppc_2000)
dd <- d[complete.cases(d$rgdppc_2000), ]
dd <- as_tibble(dd)
```

  - one perculiarity is how the GDP of a country is associated to the
    ruggedness of the terrain.
      - the association is opposite for African and non-African
        countries

<!-- end list -->

``` r
dd %>%
    mutate(is_africa = ifelse(cont_africa == 1, "Africa", "non-Africa")) %>%
    ggplot(aes(x = rugged, y = log_gdp)) +
    facet_wrap(~ is_africa, nrow = 1, scales = "free") +
    geom_smooth(method = "lm", formula = "y ~ x", color = "black", lty = 2) +
    geom_point(color = "lightblue4")
```

![](ch7_interactions_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

### 7.1.1 Adding a dummy variable doesn’t work

  - two models to begin with:
    1.  linear regression of log-GDP on ruggedness
    2.  the same model with a dummy variable for the African nations

<!-- end list -->

``` r
m7_3 <- quap(
    alist(
        log_gdp ~ dnorm(mu, sigma),
        mu <- a + bR*rugged,
        a ~ dnorm(8, 100),
        bR ~ dnorm(0, 1),
        sigma ~ dunif(0, 10)
    ),
    data = dd
)

m7_4 <- quap(
    alist(
        log_gdp ~ dnorm(mu, sigma),
        mu <- a + bR*rugged + bA*cont_africa,
        a ~ dnorm(8, 100),
        bR ~ dnorm(0, 1),
        bA ~ dnorm(0, 1),
        sigma ~ dunif(0, 10)
    ),
    data = dd
)

precis(m7_3)
```

    ##              mean         sd       5.5%     94.5%
    ## a     8.513367248 0.13533686  8.2970728 8.7296617
    ## bR    0.002812495 0.07634323 -0.1191987 0.1248237
    ## sigma 1.163054440 0.06307531  1.0622479 1.2638610

``` r
precis(m7_4)
```

    ##              mean         sd       5.5%       94.5%
    ## a      9.01583987 0.12505582  8.8159765  9.21570322
    ## bR    -0.06479984 0.06337639 -0.1660875  0.03648787
    ## bA    -1.43051778 0.16128018 -1.6882747 -1.17276091
    ## sigma  0.95759964 0.05194951  0.8745743  1.04062499

``` r
compare(m7_3, m7_4)
```

    ##          WAIC      SE    dWAIC      dSE    pWAIC       weight
    ## m7_4 476.2639 15.2637  0.00000       NA 4.328950 1.000000e+00
    ## m7_3 539.5791 13.3171 63.31523 15.05468 2.682669 1.783499e-14

  - plot the posterior distribution mean and intervals for African
    countries and the rest
      - we can see that the WAIC for the model with the dummy variable
        was lower because African countries tend to have lower GDP, not
        because it fit the different slope

<!-- end list -->

``` r
rugged_seq <- seq(-1, 8, 0.25)

mu_notafrica <- link(m7_4, data = data.frame(cont_africa = 0, 
                                             rugged = rugged_seq))
mu_africa <- link(m7_4, data = data.frame(cont_africa = 1, 
                                          rugged = rugged_seq))

mu_notafrica_mean <- apply(mu_notafrica, 2, mean)
mu_notafrica_pi <- apply(mu_notafrica, 2, PI, prob = 0.97) %>% pi_to_df()

mu_africa_mean <- apply(mu_africa, 2, mean)
mu_africa_pi <- apply(mu_africa, 2, PI, prob = 0.97) %>% pi_to_df()

bind_rows(
    tibble(cont_africa = "not Africa", 
           rugged = rugged_seq, 
           mu = mu_notafrica_mean) %>%
        bind_cols(mu_notafrica_pi),
    tibble(cont_africa = "Africa", 
           rugged = rugged_seq, 
           mu = mu_africa_mean) %>%
        bind_cols(mu_africa_pi)
) %>%
    ggplot(aes(x = rugged)) +
    geom_ribbon(aes(ymin = x2_percent, ymax = x98_percent, fill = cont_africa), 
                alpha = 0.2, color = NA) +
    geom_line(aes(y = mu, color = cont_africa), size = 1.1) +
    geom_point(data = dd, aes(y = log_gdp), size = 1, color = "lightblue4") +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    theme(legend.position = c(0.8, 0.8)) +
    labs(x = "terrain ruggedness index",
         y = "log GDP year 2000",
         color = NULL, fill = NULL,
         title = "Model of GDP by ruggedness with a dummy variable for continent",
         subtitle = "One line for in and out of Africa with the 97% interval.")
```

![](ch7_interactions_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### 7.1.2 Adding a linear interaction does work

  - we have just used the model below:

\[
Y_i \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \beta_R R_i + \beta_A A_i
\]

  - now we want to allow the relationship of \(Y\) and \(R\) to vary as
    a function of \(A\)
      - add in \(\gamma\) as a placeholder for another linear function
        that defines the slope between GDP and ruggedness
      - this is the *linear interaction effect*
      - explicitly modeling that the slope between GDP and ruggedness is
        *conditional upon* whether or not a nation is in Africa

\[
Y_i \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \gamma_i R_i + \beta_A A_i \\
\gamma_i = \beta_R + \beta_{AR}A_i
\]

  - rearranging the above forumla results in the following

\[
\mu_i = \alpha + \beta_R R_i + \beta_{AR} A_i R_i + \beta_A A_i
\]

  - fit the model using `quap()` like normal

<!-- end list -->

``` r
m7_5 <- quap(
    alist(
        log_gdp ~ dnorm(mu, sigma),
        mu <- a + gamma*rugged + bA*cont_africa,
        gamma <- bR + bAR*cont_africa,
        a ~ dnorm(8, 100),
        c(bA, bR, bAR) ~ dnorm(0, 1),
        sigma ~ dunif(0, 10)
    ),
    data = dd
)

precis(m7_5)
```

    ##             mean         sd       5.5%       94.5%
    ## a      9.1836228 0.13642358  8.9655916  9.40165401
    ## bA    -1.8460606 0.21849326 -2.1952550 -1.49686613
    ## bR    -0.1843913 0.07569178 -0.3053614 -0.06342126
    ## bAR    0.3482859 0.12750672  0.1445055  0.55206624
    ## sigma  0.9333055 0.05067821  0.8523120  1.01429910

``` r
plot(precis(m7_5))
```

![](ch7_interactions_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

  - the new model is far better than the previous two

<!-- end list -->

``` r
compare(m7_3, m7_4,  m7_5)
```

    ##          WAIC       SE     dWAIC     dSE    pWAIC       weight
    ## m7_5 469.7522 15.18797  0.000000      NA 5.359607 9.603456e-01
    ## m7_4 476.1263 15.22849  6.374185  6.1519 4.269052 3.965435e-02
    ## m7_3 539.6364 13.22767 69.884269 15.2575 2.715930 6.415809e-16

### 7.1.3 Plotting the interaction

  - nothing new, just make two plots, one for African and one for
    non-African

<!-- end list -->

``` r
rugged_seq <- seq(-1, 8, 0.25)

mu_africa <- link(m7_5, 
                  data = data.frame(cont_africa = 1, rugged = rugged_seq))
mu_africa_mean <- apply(mu_africa$mu, 2, mean)
mu_africa_pi <- apply(mu_africa$mu, 2, PI, prob = 0.97) %>% pi_to_df()

mu_notafrica <- link(m7_5, 
                     data = data.frame(cont_africa = 0, rugged = rugged_seq))
mu_notafrica_mean <- apply(mu_notafrica$mu, 2, mean)
mu_notafrica_pi <- apply(mu_notafrica$mu, 2, PI, prob = 0.97) %>% pi_to_df()

bind_rows(
    tibble(cont_africa = "not Africa", 
           rugged = rugged_seq, 
           mu = mu_notafrica_mean) %>%
        bind_cols(mu_notafrica_pi),
    tibble(cont_africa = "Africa", 
           rugged = rugged_seq, 
           mu = mu_africa_mean) %>%
        bind_cols(mu_africa_pi)
) %>%
    ggplot(aes(x = rugged)) +
    geom_ribbon(aes(ymin = x2_percent, ymax = x98_percent, fill = cont_africa), 
                alpha = 0.2, color = NA) +
    geom_line(aes(y = mu, color = cont_africa), size = 1.1) +
    geom_point(data = dd, aes(y = log_gdp), size = 1, color = "lightblue4") +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    theme(legend.position = c(0.8, 0.8)) +
    labs(x = "terrain ruggedness index",
         y = "log GDP year 2000",
         color = NULL, fill = NULL,
         title = "Model of GDP by ruggedness with an interaction term for continent",
         subtitle = "One line for in and out of Africa with the 97% interval.")
```

![](ch7_interactions_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

### 7.1.4 Interpreting the interaction estimate

  - helpful to plot implied predictions
  - often only numbers are reported, though they are difficult to
    interpret because:
      - the parameters have different meanings because they are no
        longer independent
      - it is very difficult to propagate the uncertainty when trying to
        understand multiple parameters simultaneously
  - remember that the interaction term \(y\) is a distribution
      - we can sample from it for African and non-African countries

<!-- end list -->

``` r
post <- extract.samples(m7_5)
gamma_africa <- post$bR + post$bAR*1
gamma_notafrica <- post$bR + post$bAR*0

tibble(Africa = gamma_africa,
       `not Africa` = gamma_notafrica) %>%
    pivot_longer(everything()) %>%
    ggplot(aes(x = value)) +
    geom_density(aes(fill = name, color = name), alpha = 0.3) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    labs(x = "posterior etimates of gamma",
         y = "probability density",
         title = "Probability distribution for the estimates of gamma",
         subtitle = "Gamma is the slope of the interaction term")
```

![](ch7_interactions_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

  - can use these estimates like normal:
      - e.g.: what is the probability that the slope within Africa is
        less than the slope outside of Africa

<!-- end list -->

``` r
# The probability that the slope within Africa is less than that outside.
sum(gamma_africa < gamma_notafrica) / length(gamma_africa)
```

    ## [1] 0.0031

## 7.2 Symmetry of the linear interaction

  - the interaction term we have fit has two different phrasings:
    1.  “How much does the influence of ruggedness (on GDP) depend upon
        whether the nation is in Africa?”
    2.  How much does the influence of being in Africa (on GDP) depend
        upon ruggedness?
  - the model interprets these as the same statement

7.2.1 Buridan’s interaction

  - the model’s formula can be rearranged
      - the same model can be reformulated to group the \(A_i\) terms
        together
      - shows that *linear interactions are symmetric*

\[
Y_i \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \gamma_i R_i + \beta_A A_i \\
\gamma_i = \beta_R + \beta_{AR}A_i \\
\ \\
\mu_i = \alpha + (\beta_R + \beta_{AR}A_i) R_i + \beta_A A_i \\
\mu_i = \alpha + \beta_R R_i + \beta_{AR}A_i R_i + \beta_A A_i \\
\ \\
\mu_i = \alpha + \beta_R R_i + (\beta_{AR} R_i + \beta_A) A_i \\
\]

### 7.2.2 Africa depends upon ruggedness

  - below is a plot of the reverse interpretation of the interaction
    term
      - the x-axis is now whether the country is in Africa
      - the points are the ruggedness separated by high and low (usign
        the median as the cut-off)
      - the blue slope is the expected reduction in log GDP for a
        non-rugged terrian if it was moved to Africa
      - for countries in very rugged terrains, the continent has little
        effect

<!-- end list -->

``` r
q_rugged  <- range(dd$rugged)

mu_ruggedlo <- link(m7_5,
                    data = data.frame(rugged = q_rugged[1],
                                      cont_africa = 0:1))
mu_ruggedlo_mean <- apply(mu_ruggedlo$mu, 2, mean)
mu_ruggedlo_pi <- apply(mu_ruggedlo$mu, 2, PI) %>% pi_to_df()

mu_ruggedhi <- link(m7_5,
                    data = data.frame(rugged = q_rugged[2],
                                      cont_africa = 0:1))
mu_ruggedhi_mean <- apply(mu_ruggedhi$mu, 2, mean)
mu_ruggedhi_pi <- apply(mu_ruggedhi$mu, 2, PI) %>% pi_to_df()

bind_rows(
    tibble(rugged = "low",
           cont_africa = 0:1,
           name = "low",
           mu_rugged = mu_ruggedlo_mean) %>%
        bind_cols(mu_ruggedlo_pi),
    tibble(rugged = "high",
           cont_africa = 0:1,
           mu_rugged = mu_ruggedhi_mean) %>%
        bind_cols(mu_ruggedhi_pi)
) %>%
    ggplot(aes(x = cont_africa)) +
    geom_ribbon(aes(ymin = x5_percent, ymax = x94_percent, 
                    fill = factor(rugged)), alpha = 0.2) +
    geom_line(aes(y = mu_rugged, color = factor(rugged))) +
    geom_jitter(
        data = dd, 
        aes(x = cont_africa, y = log_gdp, 
            color = ifelse(rugged < median(rugged), "low", "high")),
        alpha = 0.4, width = 0.03
    ) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1") +
    scale_x_continuous(breaks = c(0, 1), labels = c("not Africa", "Africa")) +
    labs(x = NULL,
         y  = "log GDP year 2000",
         fill = "ruggedness", color = "ruggedness",
         title = "Symmetric interpretation of the interaction term")
```

![](ch7_interactions_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

  - it is simultaneously true that:
    1.  the influence of ruggedness depends on the continent
    2.  the influence of continent depends in the ruggedness

## 7.3 Continuous interactions
