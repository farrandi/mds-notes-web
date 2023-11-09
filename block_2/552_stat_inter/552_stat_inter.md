# Statistical Inference

## Terms and Definitions

| Term                  | Definition                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| point_estimate        | A summary statistic calculated from a random sample that estimates an unknown population parameter of interest.                             |
| population            | The entire set of entities objects of interest.                                                                                             |
| population_parameter  | A numerical summary value about the population.                                                                                             |
| sample                | A collected subset of observations from a population.                                                                                       |
| observation           | A quantity or quality (or a set of these) from a single member of a population.                                                             |
| sampling_distribution | A distribution of point estimates, where each point estimate was calculated from a different random sample coming from the same population. |

- True population parameter is denoted with Greek letters. (e.g. `μ` for mean)
- Estimated population parameter is denoted with Greek letters with a hat. (e.g. `μ̂` for mean)

## Types of Statistical Questions

<img src="images/l1_stat_types.png" width="350" >

| Type        | Description                                                                                                                                                      | Example                                                                                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Descriptive | Summarizes a characteristic of a set of data without interpretation.                                                                                             | - What is the frequency of bacterial illnesses in a data set? <br> - How many people live in each US state?                                              |
| Exploratory | Analyzes data to identify patterns, trends, or relationships between variables.                                                                                  | - Do certain diets correlate with bacterial illnesses? <br> - Does air pollution correlate with life expectancy in different US regions?                 |
| Inferential | Analyzes patterns, trends, or relationships in a representative sample to quantify applicability to the whole population. Estimation with associated randomness. | - Is eating 5 servings of fruits and vegetables associated with fewer bacterial illnesses? <br> - Is gestational length different for first born babies? |
| Predictive  | Aims to predict measurements or labels for individuals. Not focused on causes but on predictions.                                                                | - How many viral illnesses will someone have next year? <br> - What political party will someone vote for in the next US election?                       |
| Causal      | Inquires if changing one factor will change another factor in a population. Sometimes design allows for causal interpretation.                                   | - Does eating 5 servings of fruits and vegetables cause fewer bacterial illnesses? <br> - Does smoking lead to cancer?                                   |
| Mechanistic | Seeks to explain the underlying mechanism of observed patterns or relationships.                                                                                 | - How do changes in diet reduce bacterial illnesses? <br> - How does airplane wing design affect air flow and decrease drag?                             |

### Performing Estimation

1. Define the population of interest.
2. Select the right sampling method according to the specific characteristics of our population of interest.
3. Select our sample size (Power Analysis).
4. Collect the sampled data.
5. Measure and calculate the sample statistic.
6. Infer the population value based on this sample statistic while accounting for sampling uncertainty.

## Sampling

| Population                     | Sample                           |
| ------------------------------ | -------------------------------- |
| $\pi_E$: population proportion | $\hat{\pi}_E$: sample proportion |
| $\mu$: population mean         | $\bar{\mu}$: sample mean         |

**Population Distribution:**

- The sample distribution is of a similar shape to the population distribution.
- The sample point estimates are identical to the values for the true population
  parameter we are trying to estimate.

**Sample Distribution of 1 Sample:**

- Taking a random sample and calculating a point estimate is a “good guess” of the unknown
  population parameter you are interested in.
- As the sample size increases:
  - the sampling distribution becomes narrower.
  - more sample point estimates are closer to the true population mean.
  - the sampling distribution appears more bell-shaped.

**Sampling Distribution of Sample Means:**

- The sampling distribution is centered at the true population mean.
- Most sample means are at or very near the same value as the true population mean.
- The sample distribution (if representative) is an estimate of the population distribution.
- The sampling distribution of the sample means is not necessarily the same shape as the
  distribution of the population distribution and tends to be more symmetrical and bell-shaped.

## Drawing samples in R (rsample::rep_sample_n)

```r
set.seed(1) # for reproducibility
sample1 <- rep_sample_n(df, size = 100, reps = 10000, replace = TRUE)
```

- default is sampling with replacement, but can be changed with `replace = FALSE`
- `size` is the number of samples to draw
- `rep` is the number of times to repeat the sampling process

## Plotting Histograms in R

```r
pop_dist <- multi_family_strata %>%
  ggplot(aes(x = current_land_value)) +
  geom_histogram(bins = 50) +
  xlab("current land value") +
  ggtitle("Population distribution")
```

# Bootstrapping

- In practical situations, we often have just one sample and struggle to estimate the sampling distribution.
- Bootstrapping, involving resampling **with replacement** from the original sample, helps create an approximation of the sampling distribution.
- This technique allows us to use our single sample as a proxy for the population and generate the necessary sampling variation for estimation.
- Basically getting more samples from 1 sample **with replacement**.

**Note:**

- Should be the same size as the original sample.

## More on Bootstrapping

- Mean of the bootstrap sample is an estimate of the sample mean not the population mean. (unlike the sampling distribution of the sample mean)
- Spread of bootstrap sample is of similar shape to the sampling distribution of the sample mean.
  - This is because we used a bootstrap sample size that was the same as the original sample size.
  - If sample size is larger than original sample: underestimate the spread.
  - This is because the empirical sample distribution is an estimate of the population distribution.

## Implementation in R

```r
set.seed(2485) # DO NOT CHANGE!

# Take a sample from the population
sample_1 <- multi_family_strata |>
    rep_sample_n(size = 10) |>

bootstrap_means_10 <- sample_1 |>
    ungroup() |> # Remove grouping
    # Removes the replicate column from rep_sample_n
    select(current_land_value) |> # Only select the column we want

    # Bootstrap from the sample 2000 times
    rep_sample_n(size = 10, reps = 2000, replace = T) |>
    # Calculate the mean for each bootstrap sample
    group_by(replicate) |>
    summarise(mean_land_value = mean(current_land_value))

bootstrap_means_10
```

# Confidence Intervals

- A plausible range of values for the population parameter
- because if we take a range of values, we can be confident that the population parameter is in that range (compared to a point estimate)

## Calculating Confidence Intervals

- One way : use the middle 95% of the distribution of bootstrap sample estimates to determine our endpoints. (2.5th and 97.5th percentiles)
- Specific to a sample, not a population
- **Confidence**: yes or no on whether the interval contains the population parameter
- Higher sample size = narrower confidence interval
  - Increasing sample size increases the precision of our estimate

### What does 95% confidence Interval mean?

- If we were to repeat this process over and over again and calculate the 95% CI many times,
  - 95% of the time expect true **population** parameter to be in the confidence interval
- The confidence level is the percentage of time that the interval will contain the true population parameter if we were to repeat the process over and over again.

- Other confidence levels: 90%, 99%
  - higher confidence level, wider the interval = more likely to contain the population parameter - higher for use cases where we need to be more confident that the interval contains the population parameter (e.g. medical trials)

Using the null distribution, the $p$-value is the area to the right of $\delta^*$ **and** to the left of $-\delta^*$. In other words, the $p$-value is doubled for the two-tailed test.
Conclusion: If we fail to reject the null hypothesis for a one-sided test, we would definitely not be able to reject it for a two-sided test.

## Steps to calculate a confidence interval

1. Calculate the true population parameter (e.g. mean), normally we don't know this

2. Get a sample from a population of size n

```r
set.seed(552) # For reproducibility.
sample <- rep_sample_n(listings, size = 40)
```

3. Look at the sample and decide on parameter of interest for distribution (e.g. median)

4. Bootstrap the sample m times (e.g. 10,000 times) with replacement from the existing sample
   and get the required statistic (e.g. median) for each bootstrap sample.

- must be the same size as the original sample

```r
set.seed(552) # For reproducibility.
bootstrap_estimates <- sample %>%
  specify(response = room_type, success = "Entire home/apt") %>%
  generate(reps = 1000, type = "bootstrap") %>%
  calculate(stat = "prop")
bootstrap_estimates
```

5. Get the decided parameter of interest (e.g., median) for each of the bootstrap samples and make a
   distribution

6. Calculate the confidence interval (e.g. 95%)
   a. use `infer::get_confidence_interval()`
   b. returns a tibble with the lower and upper bounds of the confidence interval

```r
get_confidence_interval(bootstrap_estimates, level = 0.90, type = "percentile")
```

# Hypothesis Testing

- a method of making decisions using data, whether from a controlled experiment or an observational study

## Fundamentals of Hypothesis Testing

- **Null hypothesis**: a statement of "no effect" or "no difference"
  - $H_0: p_{control} - p_{variation} = \delta = 0$, where:
    - $p_{control}$: values of the control group
    - $p_{variation}$ values from the variation group
- **Alternative hypothesis / $H_a$**: a statement of an effect or difference
  - $H_a: p_{control} - p_{variation} = \delta \neq 0$
  - claim to seek statistical evidence for
- **alpha**: the probability of rejecting the null hypothesis when it is true, typically 0.05
  - the probability of a type I error
  - the probability of a false positive
  - $\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true})$
    <br/>
- Provided a strong enough statistical evidence, we can reject the null hypothesis and accept the alternative hypothesis
- **Observed test statistic**: $\delta^*$
  - e.g. $\delta^* = \hat{m}_{chinstrap} - \hat{m}_{adelie}$
    - where $\hat{m}_{chinstrap}$ is the sample mean (estimator) body mass of Chinstrap penguins

## Framework for Hypothesis Testing (6 Steps)

1. Define your null and alternative hypotheses.

   - **Null**: The mean body mass of Chinstrap and Adelie are the same. $\mu_{Chinstrap} - \mu_{Adelie} = 0$
   - **Alt**: The mean body mass of Chinstrap and Adelie are different. $\mu_{Chinstrap} - \mu_{Adelie} \neq 0$
     - where $\mu_{Chinstrap}$ is the population mean body mass of Chinstrap penguins

2. Compute the observed test statistic coming from your original sample.

   - $\delta^* = \hat{m}_{chinstrap} - \hat{m}_{adelie}$

```r
chinstrap_adelie # data frame with chinstrap and adelie penguins only

chinstrap_adelie_test_stat <- chinstrap_adelie |>
  specify(formula = body_mass_g ~ species) |>
  calculate(
  stat = "diff in means",
  order = c("Chinstrap", "Adelie")
)
```

3. Simulate the null hypothesis being true and calculate their corresponding test statistics.

   - e.g. by randomly shuffling the data => any observed difference is due to chance
   - permutation test: randomly shuffle the data and calculate the test statistic for each permutation (Without replacement)
   - would be a normal distribution about 0 (two-tailed test)

```r
# Running permutation test
chinstrap_adelie_null_distribution <- chinstrap_adelie |>
  specify(formula = body_mass_g ~ species) |>
  hypothesize(null = "independence") |>
  generate(reps = 1000, type = "permute") |>
  calculate(
  stat = "diff in means",
  order = c("Chinstrap", "Adelie")
  )
```

4. Generate the null distribution using these test statistics.

5. Observe where the observed test statistic falls in the distribution

   - if it falls in the extreme 5% of the distribution, we reject the null hypothesis
   - i.e. if the p-value is less than $\alpha$, we reject the null hypothesis

6. If $\delta$ is near the extremes past some threshold defined with a significance level
   $\alpha$, we reject the null hypothesis. Otherwise, we fail to reject the null hypothesis.

## P-Value

- the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true
- use `get_p_value` function from `infer` R library

```r
chinstrap_adelie_p_value <- chinstrap_adelie_null_distribution|>
  get_p_value(chinstrap_adelie_test_stat, direction = "both")
```

- Results:
  - Reject Null Hypothesis if p-value < $\alpha$
  - Fail to Reject Null Hypothesis if p-value > $\alpha$

<!-- NEED TO ADD R code examples to steps -->

## Maximum Likelihood Estimation

### Definition:

- Used to FIND estimators
- **IDEA**: Given a set of data and a statistical model (assumed pdf),
  MLE finds the parameter values that make the observed data **most probable**.

#### Some key ideas:

1. Need to make distributional assumptions about the data, to specify the likelihood function
2. Need to consider nature (discrete/continuous) of the data
3. Get the parameters that maximize the likelihood function

### Likelihood Function:

- $P(dataset | params) = l(\lambda | y_1, y_2, ...)$
  - prob mass/density function of the data given the parameters
    - pdf gives density not probability (area under curve == 1)
  - generally: product of individual pdfs because iid

$$\prod_{i=1}^n f_{Y_i}(y_i| \lambda)$$

- **Context**: Used when you have a specific set of outcomes (data) and you want to understand how likely different parameter values are, given that data.

_example:_ Have a dataset of 3 obs ($y_1, y_2, y_3$) and want to know how likely it is that $\lambda = 2$. Assume _exponential_ distribution.

$$l(\lambda | y_1, y_2, y_3) = \prod_{i=1}^3 f_{Y_i}(y_i| \lambda) = \prod_{i=1}^3 \lambda e^{-\lambda y_i}$$

- in R: `dexp(y, rate = lambda, log = FALSE)`

### Objective of MLE:

- The aim is to find the parameter values that maximize this likelihood function.

$$w \in argmax\{P(dataset | params)\} = \prod_{i=1}^3 f_{Y_i}(y_i| \lambda)$$

but values are very small, so we take the log of the likelihood function to simplify the math.

$$w \in argmax\{\log(P(dataset | params))\} = \sum_{i=1}^n \ln f_{Y_i}(y_i| \lambda) $$

### Procedure:

1. Choose likelihood function
2. Take the log of the likelihood function
3. Differentiate the log-likelihood function with respect to the parameters to get max
4. Solve for the parameters

in R can do this:

```R
exp_values <- tibble(
  possible_lambdas = seq(0.01, 0.2, 0.001),
  likelihood = map_dbl(possible_lambdas, ~ prod(dexp(sample_n30$values, .))),
  log_likelihood = map_dbl(possible_lambdas, ~ log(prod(dexp(sample_n30$values, .))))
)
```

#### R optim::optimize()

- `optimize(f, interval, maximum = TRUE)`

  - `f`: function to be optimized
  - `interval`: vector of length 2 giving the start and end of the interval
  - `maximum`: logical, should the function be maximized?

  e.g.

  ```R
  # Log-likelihood function
  LL <- function(l) log(prod(dexp(sample$values, l)))

  optimize(LL, c(0.01, 0.2), maximum = TRUE)
  ```

### Properties of MLE:

- **Consistency**: As the sample size grows, the MLE converges in probability to the true parameter value.
- **Asymptotic Normality**: For many models, as the sample size grows large, the distribution of the MLE approaches a normal distribution.
- **Efficiency**: Among the class of consistent estimators, MLE often has the smallest variance (is the most efficient) under certain regularity conditions.
- MLE can be biased, but it is asymptotically unbiased (i.e. as the sample size increases, the bias goes to 0)

### Limitations:

- Requires a specified model for the underlying data distribution. If the model is incorrect, MLE can give biased estimates.
- Computation can be challenging, especially for complex models or when there's no closed-form solution.

## Central Limit Theorem

### Population and Sampling

- Lets say population of N samples, each with mean $\mu$ and standard deviation $\sigma$.

- Get a sample of n iid random samples $X_1, X_2, ..., X_n$ from the population.

$$ \bar{X} = \frac{1}{n} \sum*{i=1}^n X_i $$
$$ S^2 = \frac{1}{n-1} \sum*{i=1}^n (X_i - \bar{X})^2 $$

- n-1 is to make it unbiased

### Definition:

$\bar{X} \dot\sim N(\mu, \frac{\sigma^2}{n})$ as $n \rightarrow \infty$

- The sampling distribution of the **sample mean** of a population distribution converges to a normal distribution with mean $\mu$ and standard deviation $\frac{\sigma}{\sqrt{n}}$ as the sample size $n$ gets larger when we **sample with replacement**.
  <br/>
- NOTE: _standard deviation of the sampling distribution_ of the sample mean is called the **standard error** of the sample mean.

### Assumptions

1. The sample size is large enough (at least 30)
   - unless the population is normal
2. The sample is drawn from a population with a finite variance and mean.
3. The samples are iid (independent and identically distributed) from the population.
4. Each category in the population should have a sufficient number of samples. (e.g. #success at least 10)

When we do inference using the CLT, we required a large sample for two reasons:

- The sampling distribution of $\bar{X}$ tends to be closer to normal when the sample size is large.
- The calculated standard error is typically very accurate when using a large sample.

### Confidence Interval using CLT

$$ CI = Point \; Estimate \pm Margin \; of \; Error $$

$$ MoE = Z_x \times SE = Z_x \times \frac{\sigma}{\sqrt{n}} $$

where $Z_x$ is the z-score for the desired confidence level.

- 95% confidence level: $Z_x = Z_{0.025} = 1.96$
  - recal in R: `qnorm(0.975) = 1.96`

### Confidence Interval for Proportions

- sample proportion: $\hat{p} = \frac{\sum{X_i}}{n}$
  - it is a sample mean of 0 and 1
- Rule of thumb:
  - success: at least 10
  - failure: at least 10

**CI for proportions:**

$$ \hat{p} \pm Z_x \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} $$

where standard error is $\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$
since the variance of a bernoulli distribution is $p(1-p)$

## Hypothesis Testing via normal and T-testing

### Simple confidence interval Test

- If we want to check if mean of population is equal to a value
  - Can reasonably conclude that the population mean is not equal to the value if the value is not in the confidence interval
  - If the value is in the confidence interval, we cannot conclude that the population mean is equal to the value
    - NEED TO DO A HYPOTHESIS TEST

### Hypothesis testing

1. Permutation test (can work on any estimator)
2. Student's t-test (only works on sample mean)
3. Proportion test (only works on sample proportion)

Note: All hypothesis tests are done under the null hypothesis

### P-Value

- the probability of observing a test statistic equally or more extreme than the one you observed, given that the null hypothesis is true
  - small p-value: observed test statistic is unlikely under the null hypothesis
- use `get_p_value` function from `infer` R library

- Results:
  - Reject Null Hypothesis if p-value < $\alpha$
  - Fail to Reject Null Hypothesis if p-value > $\alpha$

#### R code examples

- get p-value of normal distribution: `pnorm(Z, mean = 0, sd = 1, lower.tail = FALSE)`

### Permutation Hypothesis test (6 Steps)

0. Define your estimator (mean, median, sd, etc.)

1. Define your null and alternative hypotheses. (population)

2. Compute the observed test statistic (sample) coming from your original sample.

3. Simulate the null hypothesis being true and calculate their corresponding test statistics.

   - e.g. by randomly shuffling the data => any observed difference is due to chance
   - would be a normal distribution about 0 (two-tailed test)

4. Generate the null distribution using these test statistics.

5. Observe where the observed test statistic falls in the distribution

   - if it falls in the extreme 5% of the distribution, we reject the null hypothesis
   - i.e. if the p-value is less than $\alpha$, we reject the null hypothesis

6. If $\delta$ is near the extremes past some threshold defined with a significance level
   $\alpha$, we reject the null hypothesis. Otherwise, we fail to reject the null hypothesis.

### Proportion test

1. Define test statistic $\delta = p_{control} - p_{variation}$

- Also define null and alternative hypothesis

2. Define theory-based CLT test statistic
   $$Z = \frac{\hat{p}_{control} - \hat{p}_{variation}}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_C} + \frac{1}{n_V})}}$$
   where:
   $$\hat{p} = \frac{\sum_{i=1}^{n_C} X_{iC} + \sum_{i=1}^{n_V} X_{iV}}{n_C + n_V}$$
   - numer: sample stat + effect size (difference in proportions)
   - denom: standard error of 2 sample
3. Simulates to Normal distribution with mean 0 and standard deviation

- Find p-value and see if it is less than $\alpha$
- Or find the CI and see if Z is in the CI
  - If Z is in the CI, we fail to reject the null hypothesis

#### proportion test in R

- use `prop.test()` function in R
  - `x` is the number of successes in Bernoulli trials
  - `n` is the number of trials
  - `alternative` is the alternative hypothesis
    - `less`, `greater`, `two.sided`
  - `conf.level` is the confidence level
  - `correct = FALSE` means we are using the normal approximation
    - if `correct = TRUE`, we are using the exact binomial test

```R
prop.test(x = click_summary$success, n = click_summary$n,
  correct = FALSE, alternative = "less"
)
```

### Pearson's Chi-Squared Test

- To identify whether **two categorical variables are independent or not**

Steps:

1. Make a contingency table

```R
library(janitor)

# Makes table with webpage, n, and num_success as columns
cont_table_AB <- click_through %>%
  tabyl(webpage, click_target)

```

2. Do some calulations, e.g. for 2x2 table:

$$\chi^2 = \sum_{i=1}^2 \sum_{j=1}^2 \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

### Chi-Squared Test in R

```R
chisq.test(cont_table_AB, correct = FALSE)
```

where:

### T Test

- Uses the CLT to approximate the sampling distribution of the sample mean
  - ONLY works on sample mean (because CLT only works on sample mean)
  - ONLY works on continuous data (because CLT only works on continuous data)
  - can be any distribution, because CLT makes it normal
- Basically similar to normal test but we have small sample size
- **Degrees of freedom** = n - 1 where n is the sample size

  - as df increases, the t-distribution approaches the standard normal distribution

- _Test Statistic_ = (sample mean - population mean) / standard error

  - standard error = standard deviation / sqrt(n)
  - more formally:
    $$t = \frac{\bar{x} - \mu}{s/\sqrt{n}}$$

- numerator: observed difference in sample means
- denominator: standard error of the sampling distribution of the two-sample difference in means for the sample size we collected.

#### One sample t-test

- Null hypothesis: the population mean is equal to a value $\mu = \mu_0$
- Test statistic: $$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$$
  - s : sample standard deviation
- In R use `t.test()` function
  - `x` is the sample data
  - `mu` is the value of the population mean
  - `alternative` is the alternative hypothesis
    - `less`, `greater`, `two.sided`
  - `conf.level` is the confidence level
  - `correct = FALSE` means we are using the normal approximation
    - if `correct = TRUE`, we are using the exact binomial test

#### Two sample t-test

- Null hypothesis: the population mean of two groups are equal $\delta = \mu_1 - \mu_2 = 0$
- Test statistic (if equal variance, $\frac{s_1^2}{s_2^2} < 2):

$$t = \frac{(\bar{x}_1 - \bar{x}_2) - \delta_0}{S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

- $S_p$ is the pooled standard deviation
  $$S_p = \sqrt{\frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}}$$
- degrees of freedom: $df = n_1 + n_2 - 2$
- in R, use `t.test()` function
  - `formula` is the formula for the test statistic `variable ~ group`
  - `data` is the data frame
  - `mu` hypothesis value of $\delta$
  - `alternative` is the alternative hypothesis
    - `less`, `greater`, `two.sided`
  - `conf.level` is the confidence level
  - `var.equal = TRUE` means we are assuming equal variance
    - if `var.equal = FALSE`, we are assuming unequal variance

### Comparison with simulation hypothesis test

| Test Type                 | Factor          | Impact on Distribution / Statistic           | Impact on p-value                       |
| ------------------------- | --------------- | -------------------------------------------- | --------------------------------------- |
| **Simulation Hypothesis** | Sample Size     | Increase => narrower null distribution       | Decrease                                |
|                           | Sample Variance | Increase => increase simulated null variance | Increase                                |
|                           | Effect Size     | Increase => more extreme test statistic      | Increase (more likely to reject null)   |
| **Normal/T-Testing**      | Effect Size     | In numerator of test statistic               | Increase (if other factors remain same) |
|                           | Sample Variance | Increase => increase standard error          | Increase                                |
|                           | Sample Size     | Increase => decrease standard error          | Decrease                                |

## Errors in Inference

### Types of Errors

- **Type I error**: False positive
  $ P(\text{reject } H_0 | H_0 \text{ is true}) = \alpha $
  <br/>
- **Type II error**: False negative
  $ P(\text{accept } H_0 | H_0 \text{ is false}) = \beta $

| Decision\Reality | True Condition Positive        | True Condition Negative       |
| ---------------- | ------------------------------ | ----------------------------- |
| Test Positive    | Correct (True Positive)        | Type I Error (False Positive) |
| Test Negative    | Type II Error (False Negative) | Correct (True Negative)       |

### Visual representation of errors

<img src="images/8_effect_size_plot.jpeg" width = 800>

Parameters:

- $\beta$, power = 1 - $\beta$
- $\alpha$: sets type I error rate (how often we reject the null hypothesis when it is true)
- cohen's d : effect size
  - $d = \frac{\mu_1 - \mu_2}{\sigma}$, where:
    - $\mu_1$: mean of group 1
    - $\mu_2$: mean of group 2
    - $\sigma$: standard deviation of the population
