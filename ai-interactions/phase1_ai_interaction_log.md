# Phase I – AI interaction log (AI pair)

## Summary

**What the AI did:** We used Claude throughout Phase I. It wrote the code for running CLIP with 13 prompts and 3 image versions (original, blurred, downscaled), and the code for the charts. It suggested which analyses to do (which predictions change, which classes they go to, how this relates to CLIP's confidence, superclasses) and helped interpret the results. It also fixed problems along the way, like the dataset download freezing.

**What we did:** We ran the experiments, pushed for more depth when the first results were only raw numbers, and decided which findings and charts to keep. The results were checked against other sub-team's baseline (99.7% of predictions identical). 

## Examples


### 1. Underscores in class names (rejected after testing)
**AI suggestion:** the class names in other subteam's prompts contain underscores ("aquarium_fish"), which probably hurts those classes.
**What the data showed:** the opposite for those classes. With underscores they are predicted much more often (aquarium fish 326 vs 85 times), so their own accuracy goes up, but they take images from other classes and overall accuracy drops by 2.5 points.
**Why it matters:** the AI's explanation sounded convincing and was wrong in direction. We only kept claims that were backed by numbers.

### 2. "I doubt it's enough for Phase 1" (pushed back)
**Prompt:** "I doubt it's enough for phase 1, we need to build charts, visualizations and go beyond the standard task."
**Situation:** after the experiment the AI only had raw numbers and a few quick statistics.
**Result:** it built charts and tested explanations for the patterns (confidence, magnet classes, superclasses, image degradation).

### 3. "It looks purely AI" (rejected parts)
**Prompt:** "I think the visualisations look PURELY AI"
**Situation:** the AI had produced the code for charts and HTML page and the template for this description.
**Decision:** we kept the code and data, and rewrote the texts ourselves.


## What I learned about working with the AI
- It is very fast at code and at trying many variants, so the analysis got much broader than I could have done myself within the limited time
- It makes wrong guesses with confidence, which only show up when they are tested
- It produced so much so quickly that the hard part was understanding and checking it, not producing it
- It worked much better once it could see our actual code instead of guessing
- It happened many times that AI added random files and functionalities that were never asked for, so it still required a lot of control
