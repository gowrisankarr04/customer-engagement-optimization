# customer-engagement-optimization
Multimodal sentiment-driven customer engagement optimization with reinforcement learning 
> An end-to-end AI system that combines **voice sentiment analysis**,
> **behavioral prediction**, **customer segmentation**, and
> **reinforcement learning** to recommend the optimal customer
> engagement action in real time.


> ## 📌 Project Overview

Businesses waste resources contacting customers without knowing
whether they are receptive. This project solves that by building
a multimodal AI pipeline that:

1. Detects customer **voice sentiment** (Positive / Negative)
   from call audio
2. Predicts **engagement/subscription likelihood** from behavioral
   and demographic data
3. **Segments customers** into 5 actionable groups
4. Uses a **Deep Q-Network (RL agent)** to recommend the optimal
   next action: Call, SMS, Email, Wait, or Do Not Disturb
5. Presents everything through an **interactive Streamlit dashboard**

---


## 🔄 Pipeline Architecture


🎤 Customer Voice Audio
        ↓
Phase 2: CNN-LSTM-Attention
   → Positive / Negative Sentiment Score
        ↓
Phase 3: XGBoost + LightGBM + CatBoost (Ensemble)
   → Engagement / Subscription Probability
        ↓
Phase 4: K-Means Clustering (K=5)
   → Customer Segment Assignment
        ↓
Phase 5: Deep Q-Network (DQN)
   → Optimal Action Recommendation
        ↓
Phase 6: Streamlit Dashboard
   → Live Interactive Interface


   ## 📊 Results Summary

| Phase | Model | Metric | Result |
|---|---|---|---|
| Phase 2 | CNN-LSTM-Attention | Test Accuracy | **69.74%** |
| Phase 2 | CNN-LSTM-Attention | AUC | **0.7543** |
| Phase 3 | Ensemble (XGB+LGBM+Cat) | AUC | **81.29%** |
| Phase 3 | Ensemble | Accuracy | **89.06%** |
| Phase 4 | K-Means | Silhouette Score | **0.44** |
| Phase 5 | DQN Agent | Accuracy | **95.86%** |

---

## 🧠 Phase 2 — Voice Sentiment Analysis

- **Dataset:** CREMA-D (7,535 audio files, 91 diverse speakers)
- **Labels:** Positive (HAP, NEU) / Negative (ANG, DIS, FEA, SAD)
- **Features:** Mel Spectrogram (128) + MFCC (40) + Chroma (12)
  = 180 combined bands
- **Architecture:** CNN (64→128→256) → LSTM (128→64) →
  Attention → Sigmoid
- **Key Decision:** Chose CREMA-D over TESS — TESS gave a
  misleading 99% with only 2 actors. CREMA-D's 91 diverse
  speakers give an honest, generalizable 69.74%.

---

## 📊 Phase 3 — Behavioral Engagement Model

- **Dataset:** UCI Bank Marketing Dataset (41,188 customers)
- **Models:** XGBoost + LightGBM + CatBoost (Soft Voting)
- **Key Decision:** Dropped `duration` column — call duration
  is only known AFTER the call ends (data leakage)
- **Imbalance Handling:** Class weights +
  threshold optimization (best threshold: 0.71)
- **5-Fold Cross Validation:**
  - XGBoost: 0.7951 ± 0.0077
  - LightGBM: 0.8010 ± 0.0060
  - CatBoost: 0.8032 ± 0.0063 ← best and most stable

---

## 🎯 Phase 4 — Customer Segmentation

- **Algorithm:** K-Means Clustering
- **Features:** Sentiment score + Behavioral score + Age +
  Campaign + Previous contacts + Economic indicators (11 total)
- **K=5 chosen** over mathematically optimal K=3 because it maps
  directly to 5 RL actions (silhouette difference only 0.02)

| Segment | Count | % | Behavioral Score |
|---|---|---|---|
| 🟢 Hot Lead | 1,506 | 3.7% | 0.874 |
| 🟡 Warm Lead | 3,435 | 8.3% | 0.770 |
| 🔵 Neutral | 8,589 | 20.9% | 0.447 |
| 🔴 Cold | 26,086 | 63.3% | 0.273 |
| 🔕 Do Not Disturb | 1,572 | 3.8% | 0.186 |

---

## 🤖 Phase 5 — Reinforcement Learning Agent (DQN)

- **Algorithm:** Deep Q-Network (DQN)
- **State (7 features):** sentiment_score, behavioral_score,
  age, campaign, previous, emp.var.rate, cons.conf.idx
- **Actions (5):** Call Now, Send SMS, Send Email,
  Wait & Nurture, Do Not Disturb
- **Reward:** 3-layer multimodal reward
  (base + sentiment bonus + segment-specific nuance)

### Key Stability Techniques
| Technique | Purpose |
|---|---|
| Experience Replay | Breaks correlation between consecutive samples |
| Target Network (update every 5 eps) | Prevents chasing a moving target |
| Huber Loss | Robust to outlier rewards (fixed Q-explosion) |
| Gradient Clipping (clipnorm=1.0) | Prevents catastrophic weight updates |
| Epsilon-Greedy (decay=0.9995) | Balances exploration vs exploitation |

### Critical Fix — Memorization vs Genuine Learning
Initial implementation achieved **100% accuracy** — a red flag.

**Root cause:** One-hot segment flags were included directly in
the agent's state, causing memorization not learning.

**Fix:** Removed segment leakage entirely. Added macroeconomic
features (emp.var.rate, cons.conf.idx) to help the agent
distinguish segments through genuine pattern recognition.

**Result:** **95.86% accuracy** — real, defensible learning.

### Per-Segment Accuracy
| Segment | Accuracy |
|---|---|
| 🟢 Hot Lead | 93.0% |
| 🟡 Warm Lead | 94.6% |
| 🔵 Neutral | 97.6% |
| 🔴 Cold | 95.4% |
| 🔕 Do Not Disturb | 99.9% |

---

## 🖥️ Phase 6 — Streamlit Dashboard

**4 pages:**
- 🏠 **Home** — Overview, pipeline diagram, key metrics
- 🎤 **Voice Sentiment** — Upload WAV → live CNN prediction
- 👤 **Customer Predictor** — Full pipeline live demo
  (behavioral score → segment → RL action)
- 📊 **Analytics** — Segment charts, model performance,
  RL action distribution

---

## 📦 Datasets

| Dataset | Source | Used In |
|---|---|---|
| CREMA-D | [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) | Phase 2 |
| UCI Bank Marketing | [UCI Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) | Phase 3, 4, 5 |

---

## 🔑 Key Technical Decisions

| Decision | Choice | Why |
|---|---|---|
| Voice dataset | CREMA-D (not TESS) | 91 diverse speakers — honest accuracy |
| Dropped feature | `duration` column | Data leakage — only known after call |
| Cluster count | K=5 (not K=3) | Maps to 5 RL actions directly |
| RL state | 7 continuous features | Prevents memorization |
| RL loss | Huber (not MSE) | Robust to reward outliers |

---

## ⚠️ Limitations

- **Static RL environment** — customer states don't change
  based on agent actions (production would need dynamic transitions)
- **Voice model** trained on acted audio (CREMA-D), not
  real call-center recordings

---

## 🚀 Future Scope

- Build a dynamic RL environment with real state transitions
- Fine-tune Phase 2 on real call-center audio recordings
- Deploy via Streamlit Cloud for permanent public access
- Integrate with live CRM systems for automated data flow

---

## 🏢 Business Applications

This framework works across any customer-facing industry —
only the behavioral dataset changes:

- 🏦 **Banking** — Credit card and loan outreach
- 📱 **Telecom** — Churn prevention campaigns
- 🛡️ **Insurance** — Policy renewal calls
- 🛒 **E-Commerce** — Cart abandonment recovery

---

## 👤 Author

**[ R. Gowri Shankar ]**
year - june 2026
