# Development pilot calibration

The grid is fixed before model predictions. It is an engineering probe, not a confirmatory rule.
The old study remains closed. No historical threshold changes.

Quality thresholds are 0.3 and 0.5. Presence-match thresholds are 0.7, 0.8 and 0.9.
Absence-match thresholds are 0.4, 0.6 and 0.7. Require absence below presence.

Obstruction thresholds are 0.2, 0.4 and 0.6. Confidence thresholds are one third, 0.5, 0.7 and 0.9.
The grid includes legacy score boundaries as starting points.

Use lexicographic configuration order to break remaining ties.

For each model, calibrate the full veto, confidence and image-only arms separately.
Select the lowest arrangement-weighted unsupported rate among eligible settings.
Require useful yield no more than 0.10 below that model's cached base.
Require present resolution and removal resolution each at least 0.50.

These limits guard against large losses of useful answers and neglect of one class.
Break ties using higher useful yield, then configuration order.
If no configuration qualifies, report calibration failure. Keep every result.

A missing class across calibration blocks selection. Never relax the rule after predictions.

For a class-specific mean, exclude arrangements without that class denominator.
Report each metric's eligible arrangement count and input denominator.
Keep those arrangements in all other metrics where they are eligible.

Use the full-veto selected removal predicate for the removal-only arm. Do not retune it.
All arms use the same undisputed support-label set for support metrics.
Include disputed records in factual scores and coverage.
Repeat support metrics with disputed labels treated as UNKNOWN.

The result is a development pilot. It does not establish a utility margin for a future test.

Use equal query weights within each eligible arrangement.
Use equal arrangement weights in each metric.
For exploratory intervals, draw 2,000 paired arrangement bootstrap samples.
Use seed 20261116 and sample within each environment.

Report the 2.5 and 97.5 percentiles of each declared difference.
Flag zero-event and degenerate intervals. They do not establish zero risk.
