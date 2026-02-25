## END TO END PROJECT


## CREATE A VIRTUAL ENVIRONMENT
```
conda create -p venv python==3.8
conda activate venv
```

## Install all necessary libraries
```
pip install -r requirements.txt
```
## Run
```
python app.py
```

## Resume Bullet Points
- Built an end-to-end Metro Traffic Volume prediction pipeline using Python, scikit-learn, and pandas, including ingestion, transformation, model training, and inference workflows.
- Trained and evaluated multiple regression models (Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, AdaBoost) and selected the best-performing model using R² score.
- Developed a Flask web app with form-based user input to generate real-time traffic volume predictions using serialized preprocessing and model artifacts.
