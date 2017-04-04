
### Statistical Power Analysis 

The statistical power is the probability that the test correctly rejects the null hypothesis ($H_0$) when the alternative hypothesis ($H_1$) is true. Or, more simply, the probability that the test will detect a true difference.  The power of a test is dependent on the number of samples, the size of the effect, standard deviation, test type, among other factors.  The plot below shows how the power is affected by sample size.

For this A/B test the power as a function of sample size is plotted as a blue line (effect size=0.3, $\sigma=10.2$).  This test has a total sample size of 542, the statistical power is 0.10 (see blue square and dashed lines).  Thus, we are very under-powered - 0.80 is the typically accepted value.  In order to detect an effect of 0.3 slides, we should collect around 8700 samples.  If the effect size were a value of 1.23, then our sample size of 542 would be enought to reach the 0.8 threshold (green square and dashed lines).

![Data Visualization](http://i0.wp.com/plot.ly/~mdrnmillennial/125.png?w=678&ssl=1)

