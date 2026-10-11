# ATLAS score and risk method note

All values below describe the current synthetic demonstration implementation, not a validated institutional policy.

## Scores

- **Academic score:** 70% average course marks + 20% course attendance + 10% observed pass rate. Available components are reweighted when data is missing.
- **Engagement score:** 35% course attendance + 35% on-time submission rate among submitted assignments + 30% average LMS activity; available components are reweighted.
- **Placement readiness score:** 25% aptitude + 25% mock coding + 20% mock interview + 10% resume + 10% group discussion + 10% communication; available components are reweighted.
- **Student success score:** 50% academic + 20% engagement + 30% placement readiness, reweighted over available component scores.

## Risk rules

- **High academic risk:** academic score below 55, or at least two recorded backlogs and average course marks below 55.
- **Moderate academic risk:** academic score below 70, course attendance below 65%, on-time submission rate below 50%, or observed LMS activity change at or below −15; high risk takes precedence.
- **Placement risk:** probability from the bundled Logistic Regression pipeline; thresholds are High ≥65%, Moderate ≥40%, otherwise Low. This is an experimental estimate from synthetic outcomes, not an operational decision.

## Pattern and blindspot definitions

- **Course bottleneck:** eligible course average at least 5 points below the overall course-record average, subject to a minimum cohort size.
- **Silent decliners:** academic score ≥70 and observed per-week LMS activity slope ≤−1.5.
- **Hidden placement risk:** success score ≥65 and placement model risk High.
- **Asked for help, not risk-flagged:** a recorded support request while neither academic nor placement risk is Moderate/High.
- **Flagged but no feedback response:** academic or placement risk Moderate/High and zero recorded feedback responses.

These signals are descriptive and need human review. They do not establish causation, diagnose students, or prove that a particular intervention will work. Record a baseline, intervention, follow-up measure and date before judging impact.
