# Hiver SDE Intern — AI Customer Support Agent

## 1. Overview

This project builds a lightweight AI customer-support agent for **SprintCare** using real customer-support conversations from Twitter.

For an incoming customer message, the agent:

1. Classifies the message into one of 8 support intents.
2. Retrieves historically similar SprintCare conversations.
3. Drafts a reply grounded in a historical SprintCare response.
4. Decides whether the case should be auto-handled or escalated to a human, with a reason.

The main goal is not only to build the system, but to evaluate it honestly on a hand-labelled golden set while avoiding data leakage.

---

## 2. Problem Framing

For SprintCare, a useful support agent should:

* identify the customer's actual support issue rather than only reacting to emotional language;
* use previous SprintCare conversations as evidence for replies;
* avoid confidently automating risky account, billing, or cancellation situations;
* escalate when intent confidence or historical evidence is insufficient;
* make its decision explainable.

### What I chose not to build

I deliberately kept the system small and reproducible rather than building a large end-to-end LLM application.

I did not build:

* a production Twitter integration;
* a full conversational memory system;
* an LLM fine-tuning pipeline;
* automated account actions such as refunds, cancellations, or plan changes;
* a production-grade safety/compliance layer.

The focus is on a measurable prototype for intent classification, evidence retrieval, reply drafting, and escalation.

---

## 3. Dataset

### Primary Dataset

**Customer Support on Twitter**

Kaggle dataset: `thoughtvector/customer-support-on-twitter`

The dataset contains roughly 3M tweets across multiple brands and includes parent/response relationships between tweets.

### Brand Selected

**SprintCare**

I selected SprintCare because it contains a large number of customer/brand response pairs and covers several recurring support problems.

After pairing customer messages with their corresponding SprintCare replies:

* Historical paired conversations: **22,313**
* Leakage-clean historical conversations used for retrieval/training: **22,055**

The final clean history removes golden-set examples by tweet ID and also removes exact golden customer-message matches.

---

## 4. Intent Taxonomy

I defined 8 intents from recurring patterns in the SprintCare data.

| Intent                               | Description                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| Network / Connectivity               | No service, slow internet, LTE/3G problems, dropped calls, SMS/data failures |
| Billing / Payment                    | Unexpected charges, bills, fees, payments, account balance                   |
| Device / Technical Issue             | Defective phone, activation, device-specific technical problems              |
| Order / Upgrade / iPhone             | iPhone orders, upgrades, preorder, shipping, eligibility                     |
| Website / App Issue                  | Website, cart, app, chat or page loading problems                            |
| Customer Service Issue               | Long waits, poor support experience, dropped support calls, agent problems   |
| Cancellation / Switching / Retention | Leaving Sprint, cancelling, switching carriers, retention concerns           |
| General Complaint / Other            | Vague complaints where no more specific support issue can be identified      |

A key labelling principle was:

> Emotion alone is not an intent.

For example, a vague message such as "Sprint sucks" is General Complaint / Other, while a message that says Sprint charged an unexpected upgrade fee is Billing / Payment.

---

## 5. Golden Evaluation Set

I created a **200-example golden evaluation set** sampled from SprintCare customer/brand conversation pairs.

The examples were manually reviewed and labelled using the 8-intent taxonomy and an escalation label (`Human` or `Auto`).

Final golden-set distribution:

### Intent

| Intent                               |   Count |
| ------------------------------------ | ------: |
| Network / Connectivity               |      53 |
| Customer Service Issue               |      29 |
| General Complaint / Other            |      28 |
| Billing / Payment                    |      26 |
| Order / Upgrade / iPhone             |      24 |
| Device / Technical Issue             |      15 |
| Cancellation / Switching / Retention |      13 |
| Website / App Issue                  |      12 |
| **Total**                            | **200** |

### Escalation

* Human: **183**
* Auto: **17**

The golden examples are kept separate from model training and retrieval history for the final evaluation.

---

## 6. System Architecture

```text
Customer message
       |
       v
 TF-IDF + Logistic Regression
       |
       +------> Intent + confidence
       |
       v
Historical SprintCare retrieval
       |
       +------> Top similar conversations
       |
       v
Historical brand reply
       |
       +------> Draft reply
       |
       v
Escalation policy
       |
       +------> Auto / Human + reason
```

### Intent Classifier

The final classifier uses:

* TF-IDF features
* unigram + bigram features
* minimum document frequency = 2
* sublinear TF
* English stop-word removal
* Logistic Regression
* balanced class weights

The model is trained on weakly-labelled historical data generated from transparent keyword rules.

The weak labels are treated as noisy supervision, not as human ground truth.

### Historical Retrieval

For a new customer message, TF-IDF cosine similarity is used to retrieve similar historical SprintCare customer messages.

The top historical conversation provides evidence for the draft response.

### Escalation

The current policy escalates:

* Billing / Payment
* Cancellation / Switching / Retention
* low-confidence intent predictions
* cases without sufficiently similar historical evidence

The reason is returned with every escalation decision.

---

## 7. Evaluation Methodology

A major part of the work was preventing evaluation leakage.

An initial experiment accidentally allowed overlap between the golden examples and the historical corpus. That made retrieval scores artificially high.

I therefore created a clean history by removing:

1. golden tweet IDs; and
2. exact golden customer-message text.

The final evaluation uses this clean history.

This makes the reported retrieval and support-agent results much more realistic.

---

## 8. Results

### Intent Classification

| Approach              |  Accuracy |   Macro F1 |
| --------------------- | --------: | ---------: |
| Majority baseline     |     26.5% |          — |
| Retrieval baseline    |     27.0% |     21.86% |
| Keyword baseline      |     28.0% |     24.32% |
| **Clean final model** | **29.5%** | **27.03%** |

The final model improves over:

* the majority baseline by **3.0 percentage points**;
* the keyword baseline by **1.5 percentage points**;
* the retrieval baseline by **2.5 percentage points**.

The improvement is modest, which is an important result rather than something to hide.

### Escalation

On the 200-example golden set:

* Accuracy: **47.0%**
* Macro F1: **37.49%**

Actual labels:

* Human: 183
* Auto: 17

Predicted labels:

* Human: 95
* Auto: 105

This shows that the current escalation policy is too aggressive about automation and should be made substantially more conservative before production use.

### Reply Quality Diagnostic

A 30-example reply-quality review was created using five dimensions:

* Relevance
* Groundedness
* Helpfulness
* Tone
* Overall

The current **provisional AI-assisted review** produced:

| Metric       | Mean / 5 |
| ------------ | -------: |
| Relevance    |     3.23 |
| Groundedness |     3.87 |
| Helpfulness  |     2.57 |
| Tone         |     3.83 |
| Overall      |     2.63 |

These scores are **not independent human-judge results**. They are used as a diagnostic rather than claimed as evidence of human agreement.

The strongest weakness is helpfulness: historical grounding often produces a polite and related response, but not necessarily a response that solves the customer's exact problem.

---

## 9. Failure Analysis

### Failure 1 — Product/entity words dominate the actual issue

Example:

> "I have the iphone in my cart, but the cart won't load. WHY"

Ground truth: **Website / App Issue**
Prediction: **Order / Upgrade / iPhone**

The model sees "iphone" and "cart" as strong order-related signals and misses that the actual failure is the website/cart.

**Hypothesis:** TF-IDF with shallow lexical features needs more context-aware features or hierarchical intent rules.

---

### Failure 2 — Customer-service problems become order problems

Example:

> "Was on phone for 35 min for a third time and almost done ordering the new iPhone and my call gets dropped."

Ground truth: **Customer Service Issue**
Prediction: **Order / Upgrade / iPhone**

The model focuses on "ordering" and "iPhone", while the actual complaint is the dropped support call.

**Hypothesis:** The classifier needs to prioritize the customer's primary failure/action rather than product mentions.

---

### Failure 3 — Emotional language overwhelms concrete intent

Example:

> "WHO THE HELL HAD THE IDEA TO DO MAINTENANCE ON A RELEASE DAY???? STUPID!!!!!!!!"

Ground truth: **Website / App Issue**
Prediction: **General Complaint / Other**

The emotional wording is strong enough to pull the classifier toward the generic complaint class.

**Hypothesis:** Intent classification should separate emotion/sentiment from the underlying support event.

---

### Failure 4 — Retrieval can be superficially similar

Example:

> "Don't go to #Sprint. They really suck."

The intent is correctly classified as General Complaint / Other, but a high lexical similarity does not necessarily mean that the retrieved historical response represents the right resolution strategy.

**Hypothesis:** Retrieval needs semantic similarity and possibly intent-aware filtering rather than relying only on TF-IDF cosine similarity.

---

### Failure 5 — Escalation policy is poorly calibrated

The golden set contains 183 Human and only 17 Auto examples, while the current system predicts 105 Auto and 95 Human.

This produces an escalation Accuracy of only 47%.

**Hypothesis:** A real support agent should use a conservative automation threshold, especially when the cost of an incorrect automated response is high. Escalation should be calibrated using precision/recall and possibly intent-specific risk thresholds rather than a single confidence threshold.

---

## 10. What Is Misleading About My Headline Number?

The headline intent accuracy is **29.5%**, but this number should not be interpreted as "the support agent is 29.5% ready for production."

There are several reasons.

### 1. The dataset is highly imbalanced

The golden set has 53 Network / Connectivity examples but only 12 Website / App Issue examples.

Accuracy can therefore hide poor performance on smaller classes.

That is why Macro F1 is also reported.

### 2. Intent boundaries are sometimes ambiguous

Some messages mention multiple issues.

For example, a message can mention an iPhone order but primarily complain about a dropped customer-service call.

Different reasonable annotators could disagree about the primary intent.

### 3. The final model only modestly beats simple baselines

The clean model reaches 29.5% accuracy compared with 28.0% for the keyword baseline.

This means the current model is learning useful signal, but the gain is small.

### 4. Reply quality is a different problem

Correct intent does not guarantee a useful response.

The provisional reply review shows an overall score of only 2.63/5, with helpfulness at 2.57/5.

### 5. Escalation is currently weak

The escalation Accuracy is 47%, and the system predicts Auto far more often than the golden labels.

Therefore, the current prototype should not be trusted to autonomously handle support at scale.

---

## 11. What I Would Do With One More Week

### 1. Improve Intent Classification

* Replace weak keyword labels with more carefully sampled human labels.
* Add a larger manually labelled training set.
* Test sentence embeddings or a small transformer model.
* Add class-specific thresholds.
* Investigate hierarchical classification: issue family first, then specific intent.

### 2. Improve Retrieval

* Use semantic embeddings rather than only TF-IDF.
* Retrieve within the predicted intent.
* Retrieve multiple historical resolutions and rank them.
* Filter out conversations where the historical response is generic or unrelated.

### 3. Improve Reply Generation

Instead of directly copying the top historical reply:

```text
Customer issue
     +
Top historical evidence
     |
     v
Controlled response template / LLM
     |
     v
Specific, grounded draft reply
```

This would preserve historical grounding while making the response specific to the current customer.

### 4. Improve Escalation

Use a cost-sensitive policy:

* high-risk intents → Human by default;
* low confidence → Human;
* weak retrieval evidence → Human;
* disagreement between classifier and retrieval → Human;
* only highly confident, low-risk cases → Auto.

### 5. Complete Independent Human Evaluation

The next evaluation would use an independent human reviewer to score the same reply-quality sample.

Then I would compare human scores with the LLM judge using:

* Pearson/Spearman correlation;
* exact or near agreement;
* Cohen's kappa or weighted kappa where appropriate.

This would provide evidence for how well the automated judge agrees with a human.

---

## 12. Decision Log

1. **Selected SprintCare** because it had enough paired customer/brand conversations for a focused support-agent prototype.
2. **Used 8 intents** to keep the taxonomy small enough to evaluate while still covering recurring support issues.
3. **Separated emotion from intent** so angry language alone does not become a separate support category.
4. **Created a 200-example golden set** because the assignment requires 150–250 hand-labelled examples.
5. **Used a majority baseline** to establish the minimum useful accuracy.
6. **Used a keyword baseline** because it is transparent and easy to reproduce.
7. **Used TF-IDF retrieval** as a simple evidence-retrieval baseline.
8. **Used weak supervision for training** because manually labelling all historical conversations would be impractical for this take-home.
9. **Used class-balanced Logistic Regression** because the intent classes are imbalanced.
10. **Removed golden IDs from historical data** to prevent direct evaluation leakage.
11. **Removed exact golden messages as well as IDs** because duplicate customer messages could otherwise leak evaluation information.
12. **Retrieved historical brand replies** rather than inventing unsupported resolutions.
13. **Escalated billing and cancellation cases** because incorrect automated handling can have higher customer impact.
14. **Reported Macro F1 alongside accuracy** because the golden classes are imbalanced.
15. **Reported provisional reply-review scores honestly** instead of presenting AI-assisted scores as independent human agreement.

---

## 13. Repository Structure

```text
hiver-sde-intern-assignment/
│
├── data/
│   ├── twcs.csv
│   ├── sprintcare_pairs.csv
│   ├── sprintcare_history_clean.csv
│   ├── sprintcare_training_clean.csv
│   ├── golden_200.csv
│   └── golden_200_reviewed.csv
│
├── src/
│   ├── remove_golden_from_history.py
│   ├── create_clean_history.py
│   ├── prepare_clean_training_data.py
│   ├── train_clean_intent_model.py
│   ├── evaluate_clean_intent.py
│   ├── evaluate_keyword_baseline.py
│   ├── evaluate_retrieval_intent.py
│   ├── evaluate_escalation_baseline.py
│   ├── evaluate_clean_support_agent.py
│   ├── evaluate_reply_quality.py
│   ├── create_human_review.py
│   ├── fill_human_review.py
│   └── support_agent.py
│
├── results/
│   ├── clean_support_agent_evaluation.csv
│   ├── clean_golden_predictions.csv
│   ├── keyword_baseline_predictions.csv
│   ├── retrieval_intent_predictions.csv
│   ├── escalation_baseline_predictions.csv
│   └── human_review_30_scored.csv
│
└── README.md
```

---

## 14. Setup

### Requirements

* Python 3.10+
* pandas
* scikit-learn
* joblib

### Create Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas scikit-learn joblib
```

### Dataset

Place the Kaggle Customer Support on Twitter dataset at:

```text
data/twcs.csv
```

---

## 15. Reproduce the Main Pipeline

From the repository root:

```powershell
python src/create_clean_history.py
python src/prepare_clean_training_data.py
python src/train_clean_intent_model.py
python src/evaluate_clean_support_agent.py
```

For baseline evaluation:

```powershell
python src/evaluate_keyword_baseline.py
python src/evaluate_retrieval_intent.py
python src/evaluate_escalation_baseline.py
```

The resulting metrics are written under:

```text
results/
```

The interactive support agent can be run with:

```powershell
python src/support_agent.py
```

Then enter a customer message, for example:

```text
My internet has been down all day and I have no LTE service.
```

Type `exit` to stop.

---

## 16. Reproducibility Notes

The final reported evaluation uses:

* a fixed 200-example golden set;
* deterministic random seed where sampling is used;
* a clean historical corpus with golden examples removed;
* saved model/vectorizer artifacts;
* saved prediction and evaluation files.

The headline result should therefore be reproducible from the repository rather than relying on an external manual calculation.

---

## 17. Limitations

This is a prototype, not a production support system.

The largest limitations are:

* weakly supervised training labels;
* relatively small golden evaluation set;
* class imbalance;
* lexical rather than semantic retrieval;
* limited reply-generation sophistication;
* weak escalation calibration;
* no independent LLM-judge vs human agreement study yet.

These limitations are intentionally reported because the assignment values proof and honest failure analysis over an inflated headline number.

---

## 18. Submission

The final repository should contain the runnable code, data-processing scripts, evaluation outputs, golden-set labelling artefacts, and this README.

Submission is made through the Hiver Notion form with the repository link.
"""

print("README text ready.")
print("Paste this into README.md in your project root.")
