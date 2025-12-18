# NHL Prediction Model - Revision Plan

---

## Section 1: Client's Complete Instructions

### Original Job Posting (Instructions.docx)

> NHL Game Prediction Model - Data Analytics
>
> Would like a NHL betting model/game outcome prediction model using Poisson & Monte Carlo, open to using others too. Would like it to go off of past two seasons plus current season weighing more. Would like games to load automatically each day, as well as the odds and O/U and pickling. After choosing the goalie from a drop down it would then compute a score. Would like home/away and season stats for this year but the past two years just using the season average stats. Want to use stats from Natural Stat Trick, MoneyPuck or Hockey Reference and have that in Apps Script so I can run it each day. Want to use stats like: goals for, shots for, CF%, xGF, xGA, power play %, penalty kill %, shooting %, face off win %, goalie stats like save %, goals saved above expected, even strength save %, using MoneyPuck for their 60 mins stats to determine if the 60 min moneyline is a good play. Also open to suggestions on stats and modeling formulas.

### Follow-up Questions from Tyler

1. "Have you made a sports model or worked with NHL before?"
2. "How long do you estimate this to take you?"
3. "Are you familiar with machine learning and making this into a machine learning model?"
4. "I would like to switch this to a Vercel platform and not Google Sheets if you could?"

### Additional Requirements (IMG_2260.jpeg - Previous Developer's Specs)

The Vercel version should include:
- Full game predicted score
- Win probability + 60-min ML probability
- Projected total vs. book total
- **1st period projected score & O/U**
- Daily games auto-loaded
- Drop-down goalie selection with heavier weighting on goalie stats
- Data from MoneyPuck / NHL API / Natural Stat Trick / Hockey Reference
- Per-period goal rates from API for 1st period modeling

### Additional Requirements (IMG_2261.jpeg - Tyler's Message)

> "Yes and then also would you be able to scrape covers.com and maybe make each game a clickable box or have an expand button to show like each teams last 5 games, their records and the last 5 H2H matchups. Similar to what covers.com offers some insight?"
>
> "Just thinking to maybe if its not able to be calculated but have a Back2Back clickable button for each team or have a way to factor it in but if a team is on a B2B it deducts some? Your thoughts?"

### Additional Requirements (IMG_2262.jpeg - Tyler's Message)

> "...I got a couple questions to see if you could add on to the cards of each game? It's 2 betting markets that my clients have shown interest in. And that's GIFT (goal in first ten mins) and 1+ SOG in first 2 mins?"
>
> "Maybe like a probability of it hitting? Is the best way to go about it. I know oddshark.com has a table for the gift prop bet showing each teams goals in the 10 mins and how many goals they've allowed"
>
> "The first 2 mins wanted to ask to see if you could do that? If there's somewhere we could filter stats to show us how often it's happened for them?"

### UI Reference (IMG_1944.jpeg - NFL Picks Interface)

Tyler shared a screenshot of an NFL picks interface showing:
- "Run End-To-End Model" button
- "Generate Picks" button
- "Export PDF" / "Export Overview PDF" buttons
- Game matchup chips (clickable buttons for each game)
- Dark theme UI
- API running on localhost:4000
- Win% legend: star = XGBoost model probability; plain = edge-derived estimate

### Latest Message from Tyler

> "So I had a guy going to make it but has bailed on me. But I'll send you my full ideas in the pictures. Maybe a card style that can be expandable with more info inside. But the cards containing the upcoming games"

---

## Section 2: Critical Issue - Deployment Platform (MUST CLARIFY FIRST)

### The Problem

The client asked to "switch this to a Vercel platform," but **Vercel does not natively support Streamlit** (the current UI framework). Vercel is optimized for static/serverless sites (Next.js, React). This is the most significant ambiguity in the project and must be resolved before proceeding.

The client likely saw the NFL UI reference (a modern web app) and associated "Vercel" with that look and feel. He may be expecting a Next.js/React front-end.

### Two Options for Tyler

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Streamlit + Streamlit Cloud** | Keep Python/Streamlit UI, deploy on Streamlit Cloud (free) | Faster to complete, works now, lower cost | Not "Vercel platform," may feel less polished than NFL reference |
| **B. Next.js + Vercel** | Rebuild frontend in Next.js/React, FastAPI backend on Vercel serverless | True Vercel deployment, matches NFL screenshot, modern UI/UX | **Complete frontend rewrite**, significant additional time and cost |

### Recommendation

- **For MVP:** Deploy current Streamlit app on Streamlit Cloud. Gets Tyler a working tool fast.
- **For Production:** If Vercel is required, rewrite frontend in Next.js calling the FastAPI backend.

### Question for Tyler (Priority #1)

> "Quick clarification on deployment: When you said 'Vercel platform,' did you mean:
>
> **Option A:** A working prediction tool deployed online (I can have this ready on Streamlit Cloud now - free, works immediately)
>
> **Option B:** A modern web app like that NFL screenshot, built in React/Next.js on Vercel (this would require rebuilding the entire frontend - additional scope)
>
> Which do you prefer? This determines the development path."

---

## Section 3: What Has Been Built

### Project Structure

```
nhl-predictor/
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── agents/
│   ├── __init__.py
│   ├── poisson_engine.py        # Poisson goal prediction model
│   ├── monte_carlo.py           # 10K game Monte Carlo simulation
│   ├── feature_engineer.py      # 3-season weighted feature engineering
│   ├── edge_calculator.py       # Kelly criterion, EV, edge detection
│   ├── ml_ensemble.py           # XGBoost/LightGBM ensemble (stubbed)
│   └── data_ingestor.py         # Data fetching (STUBBED - uses sample data)
├── api/
│   └── main.py                  # FastAPI backend (Vercel-ready)
├── models/
│   ├── __init__.py
│   └── schemas.py               # Pydantic data models
├── streamlit_app.py             # Streamlit UI (current version)
├── demo.py                      # CLI demo script
├── requirements.txt             # Python dependencies
├── vercel.json                  # Vercel deployment config
├── .gitignore
└── ARCHITECTURE.md              # System architecture documentation
```

### Completed Components

#### 1. Poisson Engine (`agents/poisson_engine.py`)
- Calculates expected goals (λ) for home and away teams
- Inputs: GF/60, GA/60, goalie GSAX, goalie SV%
- Home ice advantage factor (6%)
- Goalie adjustment based on GSAX and SV% vs league average
- Score probability matrix generation
- Over/under probability calculation
- **Status: Algorithm complete, unit-tested with sample data**

#### 2. Monte Carlo Simulator (`agents/monte_carlo.py`)
- 10,000 game simulations per prediction
- Poisson-distributed goal sampling
- Overtime/shootout resolution logic
- 60-minute (regulation) result probabilities
- Final result probabilities (including OT/SO)
- Score distribution output
- Most likely score calculation
- **Status: Algorithm complete, unit-tested with sample data**

#### 3. Feature Engineer (`agents/feature_engineer.py`)
- 3-season weighted averaging (50% current, 30% prior, 20% two years ago)
- Home/away split adjustments
- Recent form calculation (last 10 games)
- Rest/back-to-back factors
- Goalie quality index calculation
- Feature normalization for ML
- **Status: Algorithm complete**

#### 4. Edge Calculator (`agents/edge_calculator.py`)
- American odds to implied probability conversion
- Vig removal from betting lines
- Edge calculation (model prob - implied prob)
- Kelly criterion bet sizing (with 1/4 Kelly safety)
- Expected value per $100 calculation
- Confidence tiers (STRONG/MODERATE/WEAK/NO_EDGE)
- Moneyline and O/U analysis
- **Status: Algorithm complete, unit-tested with sample data**

#### 5. ML Ensemble (`agents/ml_ensemble.py`)
- XGBoost/LightGBM classifier structure
- Stacking meta-learner
- Feature importance extraction
- Model save/load functionality
- **Status: Structure only (stub), not trained**
- **Note:** NHL outcomes are inherently noisy; Poisson/MC often outperforms basic ML. Training requires 3-5 seasons of historical data. Consider as post-MVP enhancement.

#### 6. Data Ingestor (`agents/data_ingestor.py`)
- Team name normalization (all 32 NHL teams)
- Scraper structures for NST, MoneyPuck, NHL API, Odds API
- **Status: STUBBED - using sample/mock data only**
- **Critical:** This is the bottleneck. Real data integration is the most unpredictable part of the project.

#### 7. FastAPI Backend (`api/main.py`)
- All endpoints defined (`/api/games`, `/api/predict`, etc.)
- CORS configured
- **Status: Complete, Vercel serverless ready**

#### 8. Streamlit App (`streamlit_app.py`)
- Team/goalie dropdowns, betting lines input
- Results display with tabs
- **Status: Functional UI, uses sample data**

#### 9. Pydantic Schemas (`models/schemas.py`)
- All data models defined
- **Status: Complete**

### Sample Output (TOR vs MTL with sample data)

```
=== POISSON ENGINE ===
Home expected goals: 3.45
Away expected goals: 2.66
Expected Total: 6.11
Most likely score: 3-2

=== MONTE CARLO (10K sims) ===
Home win: 62.9%
Away win: 37.1%
OT rate: 15.5%

=== EDGE ANALYSIS (vs -165/+140) ===
Home ML Edge: +0.6% → PASS
Under 6.5 Edge: +7.7% → LEAN UNDER
```

### GitHub Repository

- **URL:** https://github.com/mastervb99/nhl-predictor
- **Status:** Pushed, ready for deployment

### Honest Assessment

The project is approximately **40-50% complete** toward a fully functional system:
- Core algorithms: ✅ Done
- Data integration: ❌ Stubbed (most unpredictable work)
- UI matching client vision: ❌ Needs redesign
- Prop bets: ❌ Not started

---

## Section 4: Gap Analysis

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Poisson + Monte Carlo model | ✅ Algorithm complete | Needs live data |
| 3-season weighted stats | ✅ Algorithm complete | Needs live data |
| Goalie dropdown selection | ✅ UI complete | Needs live goalie data |
| Goalie stat weighting | ✅ Algorithm complete | Needs live data |
| 60-min ML probability | ✅ Algorithm complete | None |
| Win probability | ✅ Algorithm complete | None |
| O/U probability | ✅ Algorithm complete | None |
| Betting edge analysis | ✅ Algorithm complete | None |
| Kelly criterion sizing | ✅ Algorithm complete | None |
| FastAPI backend | ✅ Complete | None |
| **Live data integration** | ❌ Stubbed | **Critical path - must be Phase 1** |
| ML ensemble (XGBoost/LightGBM) | ⚠️ Stub only | Post-MVP (requires historical data) |
| **Card-style UI** | ❌ Not built | Full rebuild needed |
| **Expandable game cards** | ❌ Not built | New feature |
| **Daily games auto-load** | ⚠️ Stubbed | Part of live data |
| **1st period predictions** | ❌ Not built | New model needed |
| **Team form (L5, H2H)** | ❌ Not built | New data source |
| **Back-to-back detection** | ⚠️ Algorithm exists | Needs UI + live schedule |
| **GIFT prop (first 10 min)** | ❌ Not built | New model + data source |
| **1+ SOG first 2 min prop** | ❌ Not built | Complex - play-by-play parsing |
| **Export PDF** | ❌ Not built | New feature |

---

## Section 5: Data Source Implementation Details

| Source | Data | Method | Feasibility | Notes |
|--------|------|--------|-------------|-------|
| **NHL API** | Schedule, scores, game logs | REST API (nhle.com) | ✅ Easy | Excellent unofficial APIs |
| **MoneyPuck** | xG, GSAX, goalie stats | CSV downloads | ✅ Feasible | Updated regularly |
| **Natural Stat Trick** | CF%, SCF%, per-60 | Scrape/CSV export | ✅ Feasible | No robust API |
| **Hockey Reference** | Historical, records | Scrape | ✅ Feasible | |
| **The Odds API** | Live betting lines | REST API | ✅ Reliable | Paid tiers |
| **Covers.com** | L5, H2H, records | Scrape | ⚠️ Risky | Blocks possible |
| **OddShark** | GIFT data | Scrape | ⚠️ Risky | ~58% league GIFT rate |
| **DailyFaceoff** | Projected goalies | Scrape | ⚠️ Risky | Often unconfirmed until game day |

### Fallback Strategies

- **Covers.com blocked:** Compute L5/H2H from NHL API game logs
- **OddShark unavailable:** Fall back to league averages (~58% GIFT)
- **1st period data unavailable:** Use ~30-35% of full game (historical average)
- **SOG data too complex:** Estimate from team shot pace

---

## Section 6: Implementation Plan (Revised Priority Order)

### V1.0 Feature Freeze

The following features constitute the V1.0 release. Any additional requests will be scoped as V1.1+.

---

### Phase 1: Live Data Integration (CRITICAL PATH)
**Scope:** Replace all stubbed data with live sources

**Estimated Effort:** 5-7 days (high uncertainty due to scraping)

**Deliverables:**
- NHL API: Daily schedule, game results
- MoneyPuck: Goalie stats (GSAX, SV%, ES SV%)
- Natural Stat Trick: Team advanced metrics (CF%, xGF, xGA)
- The Odds API: Live betting lines
- Aggressive caching layer

**Files to modify:**
- `agents/data_ingestor.py` (complete rewrite)

**Why First:** Nothing else works without real data. This is the true critical path and most unpredictable work.

---

### Phase 2: Card-Style UI Redesign
**Scope:** Replace sidebar UI with card-based layout

**Estimated Effort:** 3-4 days

**Deliverables:**
- Game cards: teams, time, venue
- Collapsed: teams + win % + predicted score
- Expanded: full details, goalie stats, betting edge
- Action buttons: Run Model, Export PDF
- Dark theme

**Files to modify:**
- `streamlit_app.py` (full rewrite)

**Note:** Can run in parallel with Phase 1.

---

### Phase 3: Back-to-Back Detection
**Scope:** B2B detection with probability adjustment

**Estimated Effort:** 1 day

**Deliverables:**
- Auto-detect B2B from schedule
- Visual B2B badge on cards
- ~5% probability reduction (algorithm exists)

---

### Phase 4: 1st Period Model
**Scope:** Period-specific predictions

**Estimated Effort:** 2-3 days

**Deliverables:**
- 1st period λ (~30-35% of full game)
- 1st period O/U probability
- 1st period predicted score

**Files to create:**
- `agents/period_model.py`

---

### Phase 5: Team Form & H2H Data
**Scope:** Last 5 games, head-to-head history

**Estimated Effort:** 2-3 days

**Deliverables:**
- Last 5 games per team
- Team records (overall, home, away)
- Last 5 H2H matchups

**Implementation:** Compute from NHL API game logs (more reliable than Covers.com)

---

### Phase 6: GIFT Prop Model
**Scope:** Goal In First 10 Minutes probability

**Estimated Effort:** 2-3 days (pending data source validation)

**Deliverables:**
- GIFT probability per game
- Team-specific first 10 min rates
- Historical frequency display

**Data:** Time-box initial investigation (1 day). If OddShark unreliable, estimate from pace stats.

---

### Phase 7: First 2 Min SOG Prop
**Scope:** 1+ Shot on Goal in first 2 minutes

**Estimated Effort:** 3-4 days (complex)

**Deliverables:**
- SOG probability per team
- Historical frequency

**Note:** Requires play-by-play parsing. May need approximations.

---

### Phase 8: PDF Export
**Scope:** Downloadable reports

**Estimated Effort:** 1-2 days

**Deliverables:**
- Single game PDF
- Overview PDF (all daily games)
- Download button

---

### Phase 9: ML Model Training & Validation (Post-MVP)
**Scope:** Train XGBoost/LightGBM on historical data

**Estimated Effort:** 5-10 days (R&D)

**Requirements:**
- Source 3-5 seasons of historical data
- Clean and feature-engineer dataset
- Train, validate, backtest
- Integrate as post-hoc adjustment to Poisson/MC

**Note:** This is a separate R&D effort. NHL outcomes are noisy; Poisson/MC often performs well without ML. Recommend as V1.1 enhancement after core system is stable.

---

## Section 7: Timeline Estimates

| Phase | Scope | Effort | Running Total |
|-------|-------|--------|---------------|
| **1** | Live Data Integration | 5-7 days | 5-7 days |
| **2** | Card UI Redesign | 3-4 days | 8-11 days (parallel with 1) |
| **3** | B2B Detection | 1 day | 9-12 days |
| **4** | 1st Period Model | 2-3 days | 11-15 days |
| **5** | Team Form (L5/H2H) | 2-3 days | 13-18 days |
| **6** | GIFT Prop | 2-3 days | 15-21 days |
| **7** | SOG Prop | 3-4 days | 18-25 days |
| **8** | PDF Export | 1-2 days | 19-27 days |
| **MVP Total** | Phases 1-8 | **4-6 weeks** | |
| **9** | ML Training (Post-MVP) | 5-10 days | +1-2 weeks |

**Note:** Estimates assume no major scraping issues. Data source instability could add 1-2 weeks.

---

## Section 8: Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [Run Model] [Generate Picks] [Export PDF]          │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ MTL@TOR │ │ VGK@COL │ │ BOS@NYR │ │ EDM@LAK │  ...      │
│  │  Card   │ │  Card   │ │  Card   │ │  Card   │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       └───────────┴───────────┴───────────┘                 │
│              [Click to expand any card]                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    PREDICTION ENGINE                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Poisson  │  │ Monte    │  │ 1st      │  │ Props    │    │
│  │ Engine   │  │ Carlo    │  │ Period   │  │ Model    │    │
│  │ ✅ Algo  │  │ ✅ Algo  │  │ 🔨 TODO  │  │ 🔨 TODO  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐                                 │
│  │ Edge     │  │ ML       │                                 │
│  │ Calc     │  │ Ensemble │                                 │
│  │ ✅ Algo  │  │ 📅 V1.1  │                                 │
│  └──────────┘  └──────────┘                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 DATA LAYER (CRITICAL PATH)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ NHL API  │  │ MoneyPuck│  │ Team Form│  │ OddShark │    │
│  │ Schedule │  │ Stats    │  │ L5/H2H   │  │ Props    │    │
│  │ 🔨 TODO  │  │ 🔨 TODO  │  │ 🔨 TODO  │  │ 🔨 TODO  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐                                 │
│  │ NST      │  │ Odds API │                                 │
│  │ Advanced │  │ Lines    │                                 │
│  │ 🔨 TODO  │  │ 🔨 TODO  │                                 │
│  └──────────┘  └──────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Section 9: Questions for Tyler (Priority Order)

### Priority #1 - MUST ANSWER BEFORE PROCEEDING

1. **Deployment platform:** When you said "Vercel platform," did you mean:
   - **Option A:** A working prediction tool deployed online (Streamlit Cloud - free, works now)
   - **Option B:** A modern React/Next.js web app on Vercel like the NFL screenshot (requires frontend rewrite)

### Priority #2 - Important Clarifications

2. **Goalie confirmation:** Default to expected starter, or require manual selection?

3. **Odds source:** Preferred API or sportsbook?

4. **PDF format:** Single page per game or condensed multi-game?

5. **GIFT/SOG data gaps:** Estimate from pace stats, or show "N/A"?

6. **Feature priority:** For V1.0, which are must-have vs. nice-to-have?
   - Core predictions (win %, O/U, score)
   - 1st period predictions
   - L5/H2H data
   - GIFT prop
   - SOG prop
   - PDF export

---

## Section 10: Risk Factors

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Covers.com blocking scraper | High | Medium | Compute L5/H2H from NHL API game logs |
| OddShark format changes | Medium | Medium | Flexible parser, fallback to league averages |
| 1st period data unavailable | Medium | Low | Fall back to ~35% of game scoring |
| NHL API rate limits | Low | Low | Aggressive caching |
| MoneyPuck CSV format changes | Low | Low | Version detection |
| 10K sims slow on hosting | Medium | Medium | Cache results, reduce to 5K |
| SOG data too complex | Medium | High | Approximate from shot pace |

### Project Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Vercel/Streamlit mismatch** | High | High | Clarify with client immediately (Question #1) |
| Continuous scope creep | Medium | Medium | V1.0 feature freeze, estimate cost of additions |
| Prop bet data unavailable | Medium | Medium | Time-box investigation (1 day), have fallback |
| Client expects ML "just works" | Medium | Medium | Separate ML to Phase 9, set expectations |

---

## Section 11: Summary

### Current State
- Core algorithms (Poisson, Monte Carlo, Edge Calculator): ✅ Complete, tested with sample data
- Data integration: ❌ Stubbed (most unpredictable work ahead)
- UI: ❌ Functional but needs redesign for client vision
- Overall: **40-50% complete** toward functional system

### Critical Path
**Live Data Integration (Phase 1)** is the bottleneck. Nothing else can be properly tested without real data.

### Biggest Risks
1. **Vercel/Streamlit mismatch** - must clarify with Tyler
2. **Scraping reliability** - data sources may block or change
3. **SOG prop complexity** - may require approximations

### Timeline
- **MVP (Phases 1-8):** 4-6 weeks
- **With ML training (Phase 9):** +1-2 weeks

### Recommended Next Steps
1. **Clarify Vercel vs Streamlit** with Tyler (Priority #1 Question)
2. **Start Phase 1** (Live Data Integration) immediately
3. **Run Phase 2** (UI Redesign) in parallel
4. **Establish V1.0 feature freeze** to manage scope
