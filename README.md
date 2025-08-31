# See you at the Movies!🎬- Oscar Prediction Model
This project shows how machine learning can predict major award wins, like the Oscars, using publicly available data such as prior wins and ratings. It's interesting not just for film fans, but also because it shows how data-driven models can uncover patterns of industry recognition, spotlight systemic biases, and even challenge the traditional notion that artistic success is purely subjective.


## ✨ Highlights
- End-to-end workflow: data integration → feature engineering → model training → threshold tuning → winner selection.
- Category-aware tweaks (e.g., **DGA** for Best Director, **PGA** for Best Picture, **SAG** for Best Actor/Best Actress) to reflect award-season momentum.
- Ensemble approach for balanced precision–recall and strong overall accuracy.

## 📊 Dataset
- For the dataset, we started by collecting award data from multiple Kaggle datasets, focusing on major shows like the Oscars, Golden Globes, SAG, and BAFTA — giving us a solid foundation of award-season context. 
- To unify this information, we built a custom integration script that linked all award scores to Oscar-nominated films, creating a timeline of each movie’s journey across the season. We manually added in DGA and PGA data, which are especially predictive for Best Director and Best Picture.
- For ratings, we used the OMDb API to fetch reliable IMDb and Rotten Tomatoes scores, since the Kaggle data was often incomplete or inconsistent.
- As we refined our model, we actually found that some features weren’t helping — specifically the Rotten Tomatoes scores. Instead of boosting our predictions, they were introducing noise and lowering overall accuracy. So, we made the call to remove them from our final model to improve performance.
- We also standardized data across the past 78 years, assigning unique movie IDs and converting all categorical data into a fully numeric format for model compatibility.


> **Path note:** The training/inference scripts expect `final_train_data_scores.csv` and `final_test_data_scores.csv` in the **working directory** (repo root unless you `cd` elsewhere).  
> The scraper expects `filtered_data.csv` in the working directory.  
> Adjust paths inside scripts if you prefer a different layout.

## 🔧 Setup

 ### 1) Environment
- Python 3.9+ recommended
- Create a virtual environment:

```
python -m venv .venv
```

## macOS/Linux:
```
source .venv/bin/activate
```
## Windows:
```
 .venv\Scripts\activate
```

### 2) Install Dependencies
```
pip install -r requirements.txt
```
### 3) Run the models after downloading the Test/Train data csv files
# XGBoost 
```python src/models/xgb.py```

# LightGBM 
```python src/models/lightgb.py```

# Voting Ensemble 
```python src/models/voting_ensemble.py```


🧪 Category-Aware Heuristics

Beyond model probabilities, category-specific boosts capture real-world award signals:

Best Picture: PGA win bump; optional IMDb ≥ 8.0 nudge.

Best Director: DGA win bump; BAFTA + Golden Globes combination bump.

Acting (Actor/Actress): Higher weight for SAG wins; “most wins” & “most nominations” signals.

📈 Results 
Ensemble achieved ~80.95% overall accuracy with F1 ≈ 0.89.

Notable gains in historically tougher categories (e.g., Best Actress).
See docs/Presentation-text.pdf for methodology, data sources, and evaluation.
