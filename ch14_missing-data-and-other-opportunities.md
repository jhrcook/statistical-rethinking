Chapter 14. Missing Data and Other Opportunities
================

  - cover two common applications of Bayesian statistics:
    1.  *measurement error*
    2.  *Bayesian imputation*

## 14.1 Measurement error

  - in the divorce and marriage data of states in the US, there was
    error in the measured variables (marriage and divorce rates)
      - this data can be incorporated into the model
      - the plot below shows the standard error in the measurement
        against median age of marriage and the population
      - seems like smaller states have more error (smaller sample size)

<!-- end list -->

``` r
# Load the data.
data("WaffleDivorce")
d <- as_tibble(WaffleDivorce) %>%
    janitor::clean_names()

p1 <- d %>%
    ggplot(aes(median_age_marriage, divorce)) +
    geom_linerange(aes(ymin = divorce - divorce_se, ymax = divorce + divorce_se),
                   size = 1, color = grey, alpha = 0.6) +
    geom_point(size = 2, color = dark_grey) +
    labs(x = "median age of marriage",
         y = "divorce rate")

p2 <- d %>%
    ggplot(aes(log(population), divorce)) +
    geom_linerange(aes(ymin = divorce - divorce_se, ymax = divorce + divorce_se),
                   size = 1, color = grey, alpha = 0.6) +
    geom_point(size = 2, color = dark_grey) +
    labs(x = "log population",
         y = "divorce rate")

p1 | p2
```

![](ch14_missing-data-and-other-opportunities_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

  - makes sense to have the more certain estimates have more influence
    on the regression
      - many *ad hoc* methods for including this confidence as a weight
        in the analysis, but they leave out some data

### 14.1.1 Error on the outcome

> **Note:** In the book, McElreath does not standardize the input
> variables for the new model (accounting for measurement error), but
> does standardize the variables in the previous model (not accounting
> for measurement error). Here I have standardized the input variables
> in both models and the differences from accounting for measurement
> error are not as astounding as reported in the book. I believe that my
> decision was correct and it does not remove the overall point of the
> section.

  - to incorporate measurement error, *replace the observed data for
    divorce rate with a distribution*
  - example:
      - use a Gaussian distribution with a mean equal to the observed
        value and standard deviation equal to the measurement’s standard
        error
      - define the distribution for each divorce rate:
          - for each observed value \(D_{\text{obs}, i}\) there will be
            one parameter \(D_{\text{est}, i}\)
          - the measruement \(D_{\text{obs}, i}\) is specified as a
            Gaussian distribution with the center of the estimate and
            standard deviation of the measurement
      - can then estimate the plausible true values consistent with the
        observed data

\[
D_{\text{obs}, i} \sim \text{Normal}(D_{\text{est}, i}, D_{\text{SE}, i})
\]

  - the model for the divorce rate \(D\) as a linear function of age at
    marriage \(A\) and marriage rate \(R\)
      - the first line is the *likelihood for estimates*
      - the second line is the linear model
      - the third line is the *prior for estimates*
      - the main difference with this model compared to the normal ones
        we have used previously is that the outcome is a vector of
        parameters
          - each outcome parameter also gets a second role as the
            unknown mean of another distribution to predict the observed
            measurement
      - information will flow in both directions:
          - the uncertainty in the measurement influences the regression
            parameters in the linear model
          - the regression parameters in the linear model influence the
            uncertainty in the measurements

\[
D_{\text{est}, i} \sim \text{Normal}(\mu_i, \sigma) \\
\mu_i = \alpha + \beta_A A_i + \beta_R R_i \\
D_{\text{obs}, i} \sim \text{Normal}(D_{\text{est}, i}, D_{\text{SE}, i}) \\
\alpha \sim \text{Normal}(0, 10) \\
\beta_A \sim \text{Normal}(0, 10) \\
\beta_R \sim \text{Normal}(0, 10) \\
\sigma \sim \text{Cauchy}(0, 2.5) \\
\]

  - a few notes on the `map2stan()` implementation of the above formula
      - turned off WAIC calculation because it will incorrectly compute
        WAIC by integrating over the `div_est` values
      - gave the model a starting point for `div_est` as the observed
        values
          - this also tells it how many parameters it needs
      - set the *target acceptance rate* from default 0.8 to 0.95 which
        causes Stan to “work harder” during warmup to improve the later
        sampling

<!-- end list -->

``` r
dlist <- d %>% 
    transmute(div_obs = divorce,
              div_sd = divorce_se,
              R = scale_nums(marriage),
              A = scale_nums(median_age_marriage))

stash("m14_1", depends_on = "dlist", {
    m14_1 <- map2stan(
        alist(
            div_est ~ dnorm(mu, sigma),
            mu <- a + bA*A + bR*R,
            div_obs ~ dnorm(div_est, div_sd),
            a ~ dnorm(0, 10),
            bA ~ dnorm(0, 10),
            bR ~ dnorm(0, 10),
            sigma ~ dcauchy(0, 2.5)
        ),
        data = dlist,
        start = list(div_est = dlist$div_obs),
        WAIC = FALSE,
        iter = 5e3, warmup = 1e3, chains = 2, cores = 2,
        control = list(adapt_delta = 0.95)
    )
})
```

    #> Loading stashed object.

  - previously, the estimate for `bA` was about -1, now it is -0.5, but
    still comfortably negative
      - including measurement error reduced the estimated effect of
        another variable

<!-- end list -->

``` r
precis(m14_1, depth = 1)
```

    #> 50 vector or matrix parameters hidden. Use depth=2 to show them.

    #>               mean        sd       5.5%      94.5%    n_eff     Rhat4
    #> a      9.556418825 0.2044546  9.2376879  9.8812416 5002.159 1.0011619
    #> bA    -1.242940436 0.3208026 -1.7527063 -0.7371894 3742.422 0.9999276
    #> bR    -0.005569283 0.3420884 -0.5595417  0.5257439 3311.623 1.0000885
    #> sigma  1.078827705 0.1961819  0.7872871  1.4068432 2199.206 1.0006505

``` r
precis(m14_1, depth = 2)
```

    #>                     mean        sd       5.5%      94.5%     n_eff     Rhat4
    #> div_est[1]  11.837344567 0.6647418 10.7887078 12.8962546  5898.515 1.0006582
    #> div_est[2]  10.845095554 1.0216492  9.2516403 12.4942308  7078.216 0.9998781
    #> div_est[3]  10.475214148 0.6233029  9.4918989 11.4958464  8184.472 0.9999431
    #> div_est[4]  12.263202297 0.8633582 10.9004805 13.6692378  7962.861 1.0001279
    #> div_est[5]   8.038420587 0.2341533  7.6635508  8.4066730  8446.959 0.9998663
    #> div_est[6]  10.860515645 0.7293194  9.6988492 12.0197682  6845.499 1.0002496
    #> div_est[7]   7.163170849 0.6414886  6.1314999  8.1759963  8157.261 0.9999701
    #> div_est[8]   8.973790825 0.8960361  7.5480498 10.4324535  8011.903 0.9997761
    #> div_est[9]   6.017474399 1.1305256  4.2243726  7.8412452  5747.723 1.0003524
    #> div_est[10]  8.564176404 0.3057308  8.0789946  9.0489278  8983.011 0.9999920
    #> div_est[11] 11.069878755 0.5229368 10.2494008 11.9252576  7799.606 1.0003111
    #> div_est[12]  8.529344782 0.9040473  7.0676997  9.9674663  6047.213 1.0000865
    #> div_est[13] 10.043196451 0.8956017  8.5872650 11.4520838  4712.640 1.0002555
    #> div_est[14]  8.097283965 0.4177928  7.4354006  8.7486496  9969.859 1.0001777
    #> div_est[15] 10.705737168 0.5345280  9.8579854 11.5586418  9068.945 0.9998466
    #> div_est[16] 10.218518886 0.7074659  9.1000986 11.3362467  8647.523 0.9998100
    #> div_est[17] 10.602823842 0.7732763  9.3709687 11.8562808  8904.705 0.9998409
    #> div_est[18] 11.997015355 0.6428376 10.9961545 13.0240463  6638.578 0.9999078
    #> div_est[19] 10.459064426 0.6867378  9.3805070 11.5720073  8404.221 1.0001903
    #> div_est[20] 10.513825734 1.0014584  8.9537905 12.1405870  4612.629 1.0001426
    #> div_est[21]  8.644745076 0.5949985  7.7052739  9.5877578  8775.663 0.9999275
    #> div_est[22]  7.656661816 0.4836718  6.8857298  8.4152966  7949.757 1.0006751
    #> div_est[23]  9.203606049 0.4757953  8.4311486  9.9581658  9816.742 0.9999509
    #> div_est[24]  7.875258330 0.5429486  7.0067771  8.7366772  7426.866 1.0003481
    #> div_est[25] 10.481750492 0.7523881  9.2793196 11.7017819  8308.266 0.9999605
    #> div_est[26]  9.644383825 0.5755285  8.7224565 10.5642970  7831.972 0.9999809
    #> div_est[27]  9.670190300 0.9201210  8.1929118 11.1134169  9571.456 1.0000392
    #> div_est[28]  9.420804254 0.7263935  8.2630031 10.5614328  8513.653 0.9997711
    #> div_est[29]  9.213878738 0.9332865  7.7619068 10.7342007  8552.353 0.9999310
    #> div_est[30]  6.398572024 0.4254509  5.7144864  7.0757319  8167.677 0.9998057
    #> div_est[31]  9.997584864 0.7735052  8.7600863 11.2435694  8810.013 0.9999900
    #> div_est[32]  6.650461344 0.2969075  6.1747721  7.1182930  8678.080 1.0001313
    #> div_est[33]  9.904728661 0.4420484  9.1884436 10.6105202  8335.879 1.0001231
    #> div_est[34]  9.498745501 0.9460678  7.9744460 10.9711598  5474.176 0.9997938
    #> div_est[35]  9.469109319 0.4088586  8.8131070 10.1293925  9625.518 0.9997665
    #> div_est[36] 12.051550732 0.7467593 10.8755220 13.2629049  7553.224 0.9999420
    #> div_est[37] 10.101744532 0.6433880  9.0870588 11.1508472  8445.059 1.0000633
    #> div_est[38]  7.816128738 0.4043816  7.1697125  8.4593735  8924.416 0.9998252
    #> div_est[39]  7.943434295 0.9903703  6.3789310  9.5326599  6432.167 1.0000883
    #> div_est[40]  8.436948246 0.5818800  7.4925961  9.3453467  9629.396 1.0003676
    #> div_est[41] 10.151963756 1.0133366  8.5469210 11.7668762  8469.031 0.9998339
    #> div_est[42] 11.053863879 0.6313468 10.0531577 12.0821468  8072.393 1.0002549
    #> div_est[43] 10.047639226 0.3319982  9.5208508 10.5844127 10446.567 0.9998101
    #> div_est[44] 11.116841635 0.7865211  9.8539162 12.3624456  5792.679 1.0000036
    #> div_est[45]  8.947189791 0.9541311  7.4564276 10.4863669  7947.125 0.9998089
    #> div_est[46]  8.959088029 0.4712254  8.2076794  9.7024021  9106.376 1.0000010
    #> div_est[47]  9.918694464 0.5616615  9.0315118 10.8059557  9935.964 0.9998099
    #> div_est[48] 10.713978276 0.8432919  9.3534651 12.0548978  9511.253 0.9997788
    #> div_est[49]  8.526858581 0.5131419  7.6963104  9.3438623  9957.818 0.9998486
    #> div_est[50] 11.131170633 1.1190173  9.2999832 12.8755595  6010.564 1.0000341
    #> a            9.556418825 0.2044546  9.2376879  9.8812416  5002.159 1.0011619
    #> bA          -1.242940436 0.3208026 -1.7527063 -0.7371894  3742.422 0.9999276
    #> bR          -0.005569283 0.3420884 -0.5595417  0.5257439  3311.623 1.0000885
    #> sigma        1.078827705 0.1961819  0.7872871  1.4068432  2199.206 1.0006505

  - the estimated effect of marriage age was reduced by including the
    measurement error of divorce
      - can see why because states with very high or low ages at
        marriage tended to have high uncertainty in divorce rates
      - in the model, the rates with greater uncertainty have been
        shrunk towards the mean that is more defined by the measurements
        with smaller measurement error  
          - this is *shrinkage*

<!-- end list -->

``` r
estimated_divorce_post <- extract.samples(m14_1)$div_est

d %>%
    mutate(estimated_divorce = apply(estimated_divorce_post, 2, mean)) %>%
    ggplot(aes(x = divorce_se, y = estimated_divorce - divorce)) +
    geom_hline(yintercept = 0, lty = 2, color = light_grey, size = 0.8) +
    geom_point(size = 2, color = dark_grey) +
    labs(x = "divorce observed standard error",
         y = "divorce estimated - divorce observed",
         title = "The more measurement error, the more shrinkage of estimated rate")
```

![](ch14_missing-data-and-other-opportunities_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

  - below, the model without accounting for measurement error is created
    and the posterior estimates of median age are shown

<!-- end list -->

``` r
stash("m14_1_noerr", depends_on = "dlist", {
    m14_1_noerr <- map2stan(
        alist(
            div_obs ~ dnorm(mu, sigma),
            mu <- a + bA*A + bR*R,
            a ~ dnorm(0, 10),
            bA ~ dnorm(0, 10),
            bR ~ dnorm(0, 10),
            sigma ~ dcauchy(0, 10)
        ),
        data = dlist,
        WAIC = FALSE,
        iter = 5e3, warmup = 1e3, chains = 2, cores = 2,
        control = list(adapt_delta = 0.95)
    )
})
```

    #> Loading stashed object.

``` r
median_age_seq <- seq(-2.3, 3, length.out = 100)
avg_marriage <- mean(dlist$R)
pred_d <- tibble(A = median_age_seq, R = avg_marriage)
m14_1_post <- link(m14_1, data = pred_d)
```

    #> [ 100 / 1000 ][ 200 / 1000 ][ 300 / 1000 ][ 400 / 1000 ][ 500 / 1000 ][ 600 / 1000 ][ 700 / 1000 ][ 800 / 1000 ][ 900 / 1000 ][ 1000 / 1000 ]

``` r
m14_1_noerr_post <- link(m14_1_noerr, data = pred_d)
```

    #> [ 100 / 1000 ][ 200 / 1000 ][ 300 / 1000 ][ 400 / 1000 ][ 500 / 1000 ][ 600 / 1000 ][ 700 / 1000 ][ 800 / 1000 ][ 900 / 1000 ][ 1000 / 1000 ]

``` r
pred_d_post <- pred_d %>%
    rename(median_age_marriage = A, marriage = R) %>%
    mutate(witherr_divorce_est = apply(m14_1_post, 2, mean),
           noerr_divorce_est = apply(m14_1_noerr_post, 2, mean)) %>%
    bind_cols(
        apply(m14_1_post, 2, PI) %>% 
            pi_to_df() %>% 
            set_names(c("with_err_5", "with_err_94")),
        apply(m14_1_noerr_post, 2, PI) %>% 
            pi_to_df() %>% 
            set_names(c("no_err_5", "no_err_94"))
    )

d %>%
    mutate(median_age_marriage = scale_nums(median_age_marriage)) %>%
    ggplot(aes(x = median_age_marriage)) +
    geom_ribbon(aes(ymin = with_err_5, ymax = with_err_94),
                data = pred_d_post,
                alpha = 0.2, fill = blue) +
    geom_ribbon(aes(ymin = no_err_5, ymax = no_err_94),
                data = pred_d_post,
                alpha = 0.2, fill = light_grey) +
    geom_line(aes(y = witherr_divorce_est), 
              data = pred_d_post,
              color = blue, lty = 2, size = 0.8) +
    geom_line(aes(y = noerr_divorce_est), 
              data = pred_d_post,
              color = grey, lty = 2, size = 0.8) +
    geom_linerange(aes(ymin = divorce - divorce_se, ymax = divorce + divorce_se), 
                   size = 0.5, color = dark_grey) +
    geom_point(aes(y = divorce), size = 0.7, color = dark_grey)
```

![](ch14_missing-data-and-other-opportunities_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

### 14.1.2 Error on both outcome and predictor
