### Key Findings
* The rating of a meal has a mixed impact on the probability of churn due to interaction with meal difficulty.
* Low difficulty meals decrease the risk of churn, whereas high difficulty increases the risk of churn, even if meals have high average ratings.
* When difficulty level is held low, average meal rating has a negative impact on churn, and higher rating lowers risk of churn.

![Data Visualization](http://mastersofmedia.hum.uva.nl/wp-content/uploads/2012/03/fry_viz2.jpg)

### Feature Definition
**Average Meal Rating** is defined as the mean rating of a meal across the customer base. In other words, instead of measuring how an individual rated a meal, every meal is assigned an average rating across the entire customer base.

![Data Visualization](http://viz.dwrl.utexas.edu/files/jesVis_0_cropped.jpg)

# Original
### Feature Definition
**Average Meal Rating** is defined as the mean meal ratings across customer base. In other words, for each customer every meal is assigned an average rating across the entire customer base as opposed to rating by that individual.

### Limitations
This analysis only considers customers who fully complete the sign-up process (as denoted in the database by user.completed_signup_at is not null). Because skips and pauses began being tracked on February 4th, 2015, we filtered the data to customers after February 4th, 2015. 

When examining the relative importance of features to the model, keep in mind that importance represents the information gain of a feature to the model and not necessarily the importance to the business.

A churned customer is defined as a paused user who's account has lapsed more than 40 days. For more information on how we defined churn versus a pause please see [this post](https://app.datascience.com/insights/definition-pause-vs-churn-f5g4d7).

Finally, feature perturbation holds all other variables equal to examine the isolated impact of a single feature on the churn model and does not consider the inter-dependence of multiple features changing for a user at once. To understand the effect of a particular set of features, further work needs to be done to ensure that the multiple variables are treated correctly and results are trustworthy. In other words, results are not additive.

### Feature Importance
The chart below displays the relative importance of features (the information gained by a particular feature) to the churn model. The yellow bar highlights the feature we're exploring while the blue bars show the remaining features. Average Meal Rating has the fourth highest level of importance to the churn model and is an early decision split for the model.

The value assigned to the feature importance of a given feature represents the information gain of a feature to the model. To explore the direction and magnitude of a specific feature's effect on churn, we turn to feature perturbation.


### Impact and Further Opportunity

When difficulty level is held low, average meal rating has a negative impact on churn, and higher rating lowers risk of churn. However, the average rating of meal shows some level of interaction with difficulty of the meal. While most of Home Chef's meal are easy to moderate, it is important to recognize the difficulty level as one of the highest-impact factors on churn. Bearing this in mind, higher-rated meals with lower difficulties are likely to reduce churn. Any meals that meet both of these criteria should be examined closely, and future analyses can explore the potential gains in marketing these meals to customers.
