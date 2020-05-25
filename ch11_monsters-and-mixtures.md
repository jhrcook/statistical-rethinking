Chapter 11. Monsters and Mixtures
================

  - build more types of models by piecing together types we have already
    learned about
      - will discuss *ordered categorical* models and
        *zero-inflated*/*zero-augmented* models
  - mixtures are powerful, but interpretation is difficult

## 11.1 Ordered categorical outcomes

  - the outcome is discrete and has different *levels* along a dimension
    but the differences between each level are not necessarily equal
      - is a multinomial prediction problem with a constraint on the
        order of the categories
      - want an estimate of the effect of a change in a predictor on the
        change along the categories
  - use a *cumulative link function*
      - the cumulative probability of a value is *the probability of
        that value or any smaller value*
      - this guarantees the ordering of the outcomes

### 11.1.1 Example: Moral intuition

  - example data come from a survey of people with different versions of
    the classic “Trolley problem”
      - 3 versions that invoke different moral principles: “action
        principle,” “intention principle,” and “contact principle”
      - the goal is how people just the different choices from the
        different principles
      - `response`: from an integer 1-7, how morally permissible is the
        action

<!-- end list -->

``` r
data("Trolley")
d <- as_tibble(Trolley)
```

### 11.1.2 Describing an ordered distribution with intercepts

  - some plots of the data
      - a histogram of the response values
      - cumulative proportion of responses
      - log-cumulative-odds of responses

<!-- end list -->

``` r
p1 <- d %>%
    ggplot(aes(x = response)) +
    geom_histogram(bins = 30) +
    labs("Distribution of response")

p2 <- d %>% 
    count(response) %>%
    mutate(prop = n / sum(n),
           cum_prop = cumsum(prop)) %>%
    ggplot(aes(x = response, y = cum_prop)) +
    geom_line() +
    geom_point() +
    labs(y = "cumulative proportion")

p3 <- d %>% 
    count(response) %>%
    mutate(prop = n / sum(n),
           cum_prop = cumsum(prop),
           cum_odds = cum_prop / (1 - cum_prop),
           log_cum_odds= log(cum_odds)) %>%
    filter(is.finite(log_cum_odds)) %>%
    ggplot(aes(x = response, y = log_cum_odds)) +
    geom_line() +
    geom_point() +
    labs(y = "log-cumulative-odds")

p1 | p2 | p3
```

![](ch11_monsters-and-mixtures_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

  - why use the log-cumulative-odds of each response:
      - it is the cumulative analog of the logit link used previously
      - the logit is the log-odds; the cumulative logit is the
        log-cumulative-odds
      - constrains the probabilities to between 0 and 1
      - this link function takes care of converting the parameter
        estimates to the probability scale
  - to use Bayes’ theorom to compute the posterior distribution of these
    intercepts, need to compute the likelihood of each possible response
    value
      - need to use the cumulative probabilities \(\Pr(y_i \ge k)\) to
        compute the likelihood \(\Pr(y_i = k)\)
      - use the inverse link to translate the log-cumulative-odds back
        to cumulative probability
      - therefore, when we observe \(k\) and need its likelihoood, just
        use subtraction:
          - the values are shown as blue lines in the next plot

\[
p_k = \Pr(y_i = k) = \Pr(y_i \le k) - \Pr(y_i \le k - 1)
\]

``` r
offset_subtraction <- function(x) {
    y <- x
    for (i in seq(1, length(x))) {
        if (i == 1) { 
            y[[i]] <- x[[i]]
        } else {
            y[[i]] <- x[[i]] - x[[i - 1]]
        }
    }
    return(y)
}

d %>% 
    count(response) %>%
    mutate(prop = n / sum(n),
           cum_prop = cumsum(prop),
           likelihood = offset_subtraction(cum_prop),
           ymin = cum_prop - likelihood) %>%
    ggplot(aes(x = response)) +
    geom_linerange(aes(ymin = 0, ymax = cum_prop), color = "grey50") +
    geom_line(aes(y = cum_prop)) +
    geom_point(aes(y = cum_prop)) +
    geom_linerange(aes(ymin = ymin, ymax = cum_prop), color = "blue",
                   position = position_nudge(x = 0.05), size = 1) +
    labs(y = "log-cumulative-odds",
         title = "Cumulative probability and ordered likelihood",
         subtitle = "Blue lines indicate the likelihood for each response.")
```

![](ch11_monsters-and-mixtures_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->
