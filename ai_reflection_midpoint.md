# AI Reflection – Phase I (midpoint)

**AI-aided pair:**
- Ivanna Pshenychna, k12148365
- [first and last name], [student ID]


## What we did with AI
We used Claude for most of the coding. It wrote the script that runs CLIP with 13 prompts and 3 image versions, the analysis script with the figures, and the interactive HTML page. It also suggested what to analyse and explained the results. We ran everything, checked the results and decided what to keep, change or remove.

## Failure patterns
- **Confident in giving wrong results.** The AI said the underscores in the class names ("aquarium_fish") probably make those classes worse. The data showed the opposite: those classes are predicted much more often, and other classes lose images instead
- **Measures that look right but aren't.** The "most unstable images" view counted how often a prompt disagrees with "a photo of a …". I noticed a sea image marked as changing under 11 of 11 prompts, although all 11 prompts answered "sea" correctly. Only the baseline was wrong. The measure never compared the prompts with each other
- **Too difficult and too polished.** The AI produced many files, long texts and a very designed HTML page. It looked obviously AI-generated, and we had to cut it down: fewer figures, simpler charts, a plain page, and texts rewritten by us
- **Problems it could not see.** The dataset download froze, and an old version of the script was still running after the AI changed it. The AI only reacted once we pasted the errors

## Best practices
- Check the AI's pipeline against the other pair's results. Our "a photo of a …" run matched other subteam's predictions on 99.7% of images
- Treat every AI explanation as a hypothesis and test it with a number
- Look at the actual images behind a pattern before believing it
- Ask for simple charts and plain output, since the defaults were more complex than needed
- Understand each figure before keeping it (we asked the AI to explain every figure with an example and kept only the simplified versions we could explain ourselves)

## Speed vs. extra work
AI made the analysis much broader than we could have done in the available time (13 prompts and 3 image versions instead of 2 prompts). But a large part of the time went into fixing setup problems, understanding what the AI had produced, checking it and removing things.
