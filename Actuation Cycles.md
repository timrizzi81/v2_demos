
This post seeks to answer the following question:
>Can we identify users who are likely candidates for time rules?

This post examines the population of users who do not use WeMo's scheduling features, yet use their device manually on a schedule. We introduce our methodology for identifying those users, and explore the types of devices they use with their WeMo products based on custom names.


### Key Findings
* Approximately 10-15% of users manually actuate their Switches, Insights, or Lightswitches consistently on a daily cycle.
* Of those users, only 40-45% have ever used time rules to actuate their devices.
* Devices in bedrooms are the most likely to be manually actuated on 24-hour cycles.
* Of devices plugged in to Switches, coffee makers are the most likely to be manually actuated on 24-hour cycles.

![Data Visualization](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=0ahUKEwiSwoWu0ovTAhXB34MKHUuvAI8QjRwIBw&url=https%3A%2F%2Fwww.promptcloud.com%2Fblog%2F6-big-data-visualization-tools-for-you&psig=AFQjCNHibyLip3Ymq_UFvsofj3E0O5frAw&ust=1491424223951858)

### Limitations
In this post, we do not consider a user's full actuation history; rather we take the last 2 weeks of their actuation data such that our findings reflect users' most recent habits. Manual actuations are examined based on hourly counts.

![Data Visualization](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=0ahUKEwie7ve70ovTAhXLz4MKHW_jCFoQjRwIBw&url=http%3A%2F%2Fnupps.olpa.com.br%2Fdata%2F&psig=AFQjCNHibyLip3Ymq_UFvsofj3E0O5frAw&ust=1491424223951858)


### Identifying Common Actuation Schedules

To determine whether actuation patterns are cyclical, we conduct statistical hypothesis tests to evaluate whether a user's actuations are significantly correlated with previous actuations for a given interval.

For instance, if a user's actuation frequency at 12:00 PM is significantly correlated with their behavior at 5:00 AM, we conclude that the user actuates their device on a 6 hour cycle. If a device is used on both a 6 hour and 12 hour cycle, we simplify by stating the user is on a 6 hour cycle, as the 12 hour cycle is encompassed by the 6 hour cycle.

Below, we plot the strength of correlation in behavior at different time intervals.

The plot below illustrates the average autocorrelation between a user's actuations at a given time and their actuations t hours ago:


We find that there are strong daily cycles of actuation patterns. While there are longer cycles, some cases of having 48 or 72 cycles may simply be artifacts of having a 24 hour cycle. Therefore, for each user with compute the strongest single cyclic pattern. For instance, if a user's 24 hour autocorrelation is .1 and their 48 hour autocorrelation function is .08, then we would determine that the strongest cycle for that user is 24 hours. We plot the frequency of strongest cycles below.

![Data Visualization](http://biomedicalcomputationreview.org/sites/default/files/u38/05-06-vibrating-nodes.png)



Our final step is to determine whether each user's strongest cycle is statistically significant (i.e. whether the results are significant enough to not be attributed to random chance). The plot below illustrates the frequency of statistically significant strongest cycles.

Most of the observed multi-day patterns are either insignificant or are artifacts of daily use, though there are a small number of users that exhibit 2-3 day cycles of usage. Overall, 24-hour cycles are the most common, with 10-15% of users actuating on this schedule.




### Users with Daily Cycles

Prevalence of daily manual actuation cycles varies between device types and locations. Bedrooms have the highest percentage of devices that follow 24-hour manual cycles. This is especially pronounced for Insights, as 19% of bedroom insights are actuated on 24-hour cycles. By contrast, outdoor Insights and Switches are the least-often actuated on 24-hour manual cycles, at 6% and 10%, respectively. Interestingly, this does not apply to outdoor Lightswitches; 17% of these devices are actuated on manual 24-hour cycles.


### Impact and Further Opportunities

This post explored the prevalence of repeated cycles of manual device actuations, with the intent of determining the viability of recommending rules based on these patterns. 24-hour cycles are common, with some device subsets utilizing them up to 19% of the time.

Of the users on 24-hour cycles, only 40-45% have ever used time rules. Because these users are using manual actuations in a pattern that lends itself to time rules, the value they are extracting from their devices could potentially be enhanced by recommending rules. This could be done via enhancements to the app, e.g. implementation of push notifications based on patterns. This analysis is also valuable as a measurement of overall use/disuse of time rules in situations where they would be helpful. Future analyses can examine the viability of recommending other rule types based on behavioral patterns.
