"""
Creation of the PEAL2 (Perceptual Evaluation of Adversarial Samples bound by L2) Dataset:

Define L2 Ranges that reflect the Evaluation-Choices in the Survey:

5: The attack is very obvious. Differences between the original and the adversarial sample are clear and easily noticeable without effort (e.g., significant artifacts or distortions).
4: The attack is clearly visible. Differences are noticeable on a quick glance, but they are less severe than those in the "5" category.
3: The attack is noticeable. You can see the differences if you compare the images normally.
2: The attack is noticeable but subtle. You can see the differences if you compare the images carefully, but they might not be immediately apparent.
1: The attack is barely perceptible. Differences are very faint and require close scrutiny to detect.
0: The attack is imperceptible. You cannot detect any differences between the original and the adversarial sample; they appear identical to the human eye.

initially set by Personal Judgement / Qualitative Analysis of FGSM samples

L2 ranges:

5: x - x
4: x - x
3: x - x
2: x - x
1: x - x
0: x - 0.0

results in 5000 samples for one class of adversarial attacks

results in 14 x 5000 -> 70000 adversarial samples grouped by average l2 score

"""

