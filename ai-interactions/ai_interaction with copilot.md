Please how we can improve the inputs to get better results?

Your script only reads actual_class and predicted_class from the CSV; it doesn’t see the original inputs or change the model. It can help identify which classes need attention:

Start with the bottom 10 classes and the “most common confusions.” If dog is often predicted as cat, review examples of both classes.
Add more correctly labeled, varied training examples for the weak or confused classes—different angles, lighting, backgrounds, sizes, and conditions.
Check for wrong labels, duplicates, or poor-quality examples. Noisy labels can make results worse.
If some classes have far fewer examples, add training examples for them or use an appropriate balancing method.
Keep a separate test set unchanged while you make improvements. Otherwise, you may make the score look better without knowing whether the model generalizes.
Compare results before and after each change using the same test examples. If you mean image inputs, tell me what the model is classifying and how you prepare the images, and I can suggest more specific improvements.

Please do it for me

I added and ran a prompt-ensemble experiment. It tries four phrasings for each class name—converting underscores to spaces—and averages their CLIP text embeddings. Your original scripts and results are unchanged.

On the same 10,000 CIFAR-100 test images:

Prompts	Accuracy
Original baseline	61.73%
Existing wording change	62.23%
New prompt ensemble	63.80%
Compared with the baseline, the new prompts corrected 695 predictions and made 488 previously correct predictions incorrect—a net gain of 207. That’s a promising improvement, not a guarantee on other images. Since this experiment uses the test set, avoid repeatedly tuning prompts against it; use separate validation examples for further trials.

Files: prompt_ensemble_results.csv and predict_prompt_ensemble.py.