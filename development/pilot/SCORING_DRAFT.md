# Development pilot: proposed model contract

This draft awaits review. It does not authorize task inference.
The pilot uses the v1 arrangement list and split in the audit ledger.

## Inputs

Resize each source image to 1200 by 675 RGB pixels with Lanczos filtering.
Draw the same red reference-box outline on every arm's reference image.
Use no box or marker in the query image.
Encode both images as PNG with no text, EXIF or other source metadata.
Pass image objects in memory. Do not pass source paths or names to the model.

The model's own processor may resize or tile these images. Record its exact settings.
Inspect the marked, resized inputs before assigning final support labels.

## Base prompt

The first image is a reference. The red box marks the remembered target.
The second image is a later view of the same scene.
The target was either left in its original region or removed from the scene.
Choose A for PRESENT, B for REMOVED, or C for UNKNOWN.

Choose PRESENT only if the image supports that the target remains in its original region.
Choose REMOVED only if the image supports absence from that region.

Choose UNKNOWN if the view does not support either decision.
A missed detection alone does not establish removal.
Earlier presence alone does not establish current presence behind a blocker.
Return exactly one answer code.

The target description is supplied in a separate fixed line: `Target: {description}.`

## Place-first prompt

Use the base prompt with this additional instruction immediately before the answer request:

First inspect the remembered region and decide which parts are visible or blocked.
Then inspect for evidence of the target and choose the supported answer code.

The model is constrained to one code. This instruction does not produce an observable reasoning trace.

## Scores

Use the model's chat template to produce the exact assistant prefix after both images and the prompt.
Compute the conditional log probability for each complete candidate continuation.
Each continuation contains the code A, B or C followed by the model's assistant-turn terminator.
Sum the conditional token log probabilities across the entire continuation, including termination.
Verify that each candidate is tokenized in the context of the exact assistant prefix.
Retain the candidate token IDs and per-token log probabilities for every call.

If a template does not provide an unambiguous terminator, stop for review.
Choose the greatest complete-continuation log probability, with A, B, C as the fixed tie order.
Normalize the three candidate scores with softmax to obtain a three-choice score.
The confidence control uses the selected answer's normalized score. These are not calibrated probabilities.

The base and place-first answers use constrained three-choice decoding, not spontaneous generation.
Apply all veto and confidence controls to the same cached base decision and scores.
A failed model call is an operational UNKNOWN with a separate failure flag.
No retry is permitted in this pilot unless a rule is approved before predictions.

## Validation before task predictions

Use synthetic images unrelated to the task to test multi-image input and complete scoring.
Test tokenization boundaries and termination for both models.

Test that all three continuations have finite scores and reproduce the cached decision.
Test that no model message contains source paths, metadata, state or pair identifiers.
Record model revision, processor settings, hashes, precision, software and timings.
If a model cannot satisfy this contract, do not substitute first-token confidence.
