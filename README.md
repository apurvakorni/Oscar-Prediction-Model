# See you at the Movies!🎬- Oscar Prediction Model
Predict winners in major Academy Award categories using gradient-boosted models and an ensemble model that blends **XGBoost**, **LightGBM**, and **CatBoost**.

## ✨ Highlights
- End-to-end workflow: data integration → feature engineering → model training → threshold tuning → winner selection.
- Category-aware tweaks (e.g., **DGA** for Director, **PGA** for Picture, **SAG** emphasis for acting) to reflect award-season momentum.
- Ensemble approach for balanced precision–recall and strong overall accuracy.


> **Path note:** The training/inference scripts expect `final_train_data_scores.csv` and `final_test_data_scores.csv` in the **working directory** (repo root unless you `cd` elsewhere).  
> The scraper expects `filtered_data.csv` in the working directory.  
> Adjust paths inside scripts if you prefer a different layout.

## 🔧 Setup

### 1) Environment
- Python 3.9+ recommended
- Create a virtual environment:

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate


### 2) Install Dependencies
# Base requirements
pip install -r requirements.txt

# If you plan to run the ensemble (uses CatBoost)
pip install -r requirements_full.txt
# or: pip install catboost
🧪 Category-Aware Heuristics

Beyond model probabilities, category-specific boosts capture real-world award signals:

Best Picture: PGA win bump; optional IMDb ≥ 8.0 nudge.

Best Director: DGA win bump; BAFTA + Golden Globes combination bump.

Acting (Actor/Actress): Higher weight for SAG wins; “most wins” & “most nominations” signals.

📈 Results (from project notes)

Ensemble achieved ~80.95% overall accuracy with F1 ≈ 0.89.

Notable gains in historically tougher categories (e.g., Best Actress).
See docs/Presentation-text.pdf for methodology, data sources, and evaluation.
