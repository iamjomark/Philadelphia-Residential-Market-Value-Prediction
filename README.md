# Toward Smarter Valuation: A Machine Learning Approach to Residential Property Mass Appraisal in Philadelphia

## Executive Summary

Philadelphia is facing a **severe housing crisis** marked by a shortage of affordable housing, rising rents, and **overvaluation of properties**. The **Office of Property Assessment (OPA)** currently uses outdated sales data and traditional statistical methods such as **Multiple Regression Analysis** for property valuations. These methods fail to capture **non-linear relationships** and struggle to keep up with rapidly changing market conditions, resulting in **mispricing** and ineffective housing policies.

In response, the **Housing Association of Philadelphia (HAP)** implemented **machine learning (ML) models** using updated property data up to 2020 to enhance the **accuracy, fairness, and transparency** of property assessments. 

---

## Key Findings

* The **Random Forest** model performed the best with:

  * **R² = 0.948** (strong explanatory power)
  * **MAE = 12,208.7**
  * **MAPE = 8.84% (<10%)**
  * **RMSE = 36,538.54**
* Only **7.5% of test predictions** were overvalued, demonstrating strong model reliability.
* **Feature importance & SHAP analysis** identified **area, location, and interior condition** as the primary factors influencing property values.

---

## Actionable Recommendations

1. **Affordable Housing Development**

   * Prioritize **high-value areas** like **Wards within and surrounding City Center and Ward 9 (Germantown/Chestnut Hill)**.
   * Expanding supply in these zones will provide better access for financially strained households.

2. **Public-Private Collaboration**

   * Promote **affordable housing units** and **flexible loan schemes** in collaboration with the private sector.
   * Guide **real estate investors** to focus on larger properties in prime areas and enhance interior quality.

3. **Policy Integration**

   * Incorporate **affordable housing into high-value neighborhoods**.
   * Use insights from ML-driven models to inform **evidence-based policy decisions**.

---

## Limitations

* **Data Constraints**: Missing values and dropped features impacted model accuracy.
* **Location Granularity**: Grouping diverse neighborhoods within wards reduced the precision of the model.
* **Computational Limits**: Limited computing resources restricted advanced model tuning.
* **Domain Expertise Gap**: Lack of prior expertise in U.S. property appraisal meant the team had to quickly adapt to local practices.

Despite these challenges, the model produced **satisfactory and actionable results**, proving that **machine learning significantly outperforms traditional methods**. Future improvements in **data granularity**, **computational resources**, and **domain expertise** will further enhance the model's accuracy and policy effectiveness.

---

## Conclusion

By adopting a **machine learning–based mass appraisal** approach, Philadelphia can achieve **more accurate property valuations**, implement **smarter housing policies**, and take significant steps toward addressing the **housing affordability crisis**.

---

## Files

* **Dataset**: `PHL_OPA_PROPERTIES.csv` (located inside the 7z archive with the same name)
* **Notebook**: [Toward Smarter Valuation: A Machine Learning Approach to Residential Property Mass Appraisal in Philadelphia](Toward%20Smarter%20Valuation%20A%20Machine%20Learning%20Approach%20to%20Residential%20Property%20Mass%20Appraisal%20in%20Philadelphia.ipynb)
* **Model Dataset**: `df_model.csv` used for modeling
* **Prediction Dataset**: `df_prediction.csv` used for analyzing prediction results (residual analysis)
* **Python Files for Streamlit**: Available for local deployment
* **Tableau Dashboard**: [Interactive Dashboard on Tableau](https://public.tableau.com/views/BetaGroup-TowardSmarterValuation_AMachineLearningApproachtoResidentialPropertyMassAppraisalinPhiladelphia/DashboardMV?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
* **App Deployment (Local)**: [Streamlit App - Local Access](https://philadelphia-housing-price-betagroup.streamlit.app/)
  *(Note: This app is unstable for public access. For better performance, use local deployment.)*

---

## How to Use the Streamlit Application (Local)

1. Clone or download all the following files into one project folder:

   * `about.py`
   * `app.py`
   * `conn.py`
   * `custom_components.py`
   * `pipelines.joblib`
   * `prediction.py`
   * `prediction_batch.py`
   * `prediction_single.py`
   * `style.css`

2. Open terminal or command prompt.

3. Navigate to the project folder:

   ```bash
   cd path/to/your/folder
   ```

4. Launch the Streamlit app by typing:

   ```bash
   streamlit run app.py
   ```

This will launch the application locally on your machine.
