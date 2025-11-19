import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
import joblib
import shap
import time
import json
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train_model(country_code):
    """
    Train XGBoost and LightGBM models to predict view velocity with improved regularization and hold-out set.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"⏳ Starting training for country {country_code}...")

    # Load data
    start_time = time.time()
    df = pd.read_csv(f'{country_code}_processed_youtube_trending.csv')
    print(f" - Loaded data: {time.time() - start_time:.2f} seconds")

    # Select features and target
    features = [
        'view_count', 'likes', 'dislikes', 'comment_count',
        'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
        'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
        'comments_per_day', 'title_length', 'title_sentiment', 'view_growth_rate'
    ]
    target = 'view_velocity'

    # Prepare dataset
    X = df[features]
    y = df[target]

    # Log transformation for skewed features
    skewed_features = ['view_count', 'likes', 'dislikes', 'comment_count', 'like_dislike_ratio', 'dislikes_per_comment', 'likes_per_day', 'comments_per_day', 'view_growth_rate']
    for feature in skewed_features:
        clean_feature = X[feature].replace([np.inf, -np.inf], np.nan).fillna(0)
        X.loc[:, feature] = np.log1p(clean_feature).astype(float)


    # Handle missing values and infinities
    full_data = X.copy()
    full_data['target'] = y
    inf_nan_report = {
        'inf_count': (full_data == np.inf).sum().to_dict(),
        'nan_count': full_data.isna().sum().to_dict()
    }
    with open(f'{output_dir}/{country_code}_inf_nan_report.json', 'w') as f:
        json.dump(inf_nan_report, f)
    full_data = full_data.replace([np.inf, -np.inf], np.nan).fillna(full_data.median())

    # Split back
    X_clean = full_data[features]
    y_clean = full_data['target']

    # Save preprocessed data
    preprocessed_data = X_clean.copy()
    preprocessed_data[target] = y_clean
    additional_columns = ['video_id', 'tags', 'title', 'publishedAt', 'trending_date', 'categoryId']
    for col in additional_columns:
        if col in df.columns:
            preprocessed_data[col] = df.loc[preprocessed_data.index, col]
    preprocessed_data.to_csv(f'{country_code}_preprocessed_youtube_trending_data.csv', index=False)
    print(f"✅ Saved preprocessed file: {country_code}_preprocessed_youtube_trending_data.csv")

    # Compute sample weights
    category_counts = df['categoryId'].value_counts()
    weights = df['categoryId'].map(lambda x: 1 / category_counts[x] if x in category_counts else 1)

    # Split train/hold-out/test
    X_temp, X_holdout, y_temp, y_holdout, weights_temp, weights_holdout = train_test_split(
        X_clean, y_clean, weights, test_size=0.1, random_state=42
    )
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X_temp, y_temp, weights_temp, test_size=0.2, random_state=42
    )

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of X_holdout: {X_holdout.shape}")

    # -------------------------- XGBoost --------------------------
    print("⏳ Training XGBoost...")
    start_time = time.time()
    param_grid_xgb = {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'lambda': [1, 2, 5],
        'alpha': [0.5, 1, 2]
    }
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42
    )
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search_xgb.fit(X_train, y_train, sample_weight=weights_train)
    print(f"Best parameters for XGBoost ({country_code}): {grid_search_xgb.best_params_}")
    model_xgb = grid_search_xgb.best_estimator_

    # Evaluate on test set
    y_pred_xgb = model_xgb.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Evaluate on hold-out set
    y_pred_xgb_holdout = model_xgb.predict(X_holdout)
    mae_xgb_holdout = mean_absolute_error(y_holdout, y_pred_xgb_holdout)
    rmse_xgb_holdout = np.sqrt(mean_squared_error(y_holdout, y_pred_xgb_holdout))
    r2_xgb_holdout = r2_score(y_holdout, y_pred_xgb_holdout)

    print(f"XGBoost Performance ({country_code}):")
    print(f"Test MAE: {mae_xgb:.6f}, RMSE: {rmse_xgb:.6f}, R2: {r2_xgb:.6f}")
    print(f"Hold-out MAE: {mae_xgb_holdout:.6f}, RMSE: {rmse_xgb_holdout:.6f}, R2: {r2_xgb_holdout:.6f}")

    # Feature importance
    plt.figure(figsize=(12, 8))  # Tăng kích thước biểu đồ
    xgb.plot_importance(model_xgb, max_num_features=10)
    plt.title(f'XGBoost Feature Importance ({country_code})')
    plt.subplots_adjust(left=0.35)  # Tăng lề trái để nhãn trục y hiển thị đầy đủ
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{country_code}_xgb_feature_importance.png')
    plt.close()

    # -------------------------- LightGBM --------------------------
    print("⏳ Training LightGBM...")
    start_time = time.time()
    param_grid_lgb = {
        'num_leaves': [20, 31, 50],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'bagging_fraction': [0.8, 1.0],
        'reg_lambda': [1, 2, 5]
    }
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        verbose=-1
    )
    grid_search_lgb = GridSearchCV(lgb_model, param_grid_lgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search_lgb.fit(X_train, y_train, sample_weight=weights_train)
    print(f"Best parameters for LightGBM ({country_code}): {grid_search_lgb.best_params_}")
    model_lgb = grid_search_lgb.best_estimator_

    # Evaluate on test set
    y_pred_lgb = model_lgb.predict(X_test)
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)

    # Evaluate on hold-out set
    y_pred_lgb_holdout = model_lgb.predict(X_holdout)
    mae_lgb_holdout = mean_absolute_error(y_holdout, y_pred_lgb_holdout)
    rmse_lgb_holdout = np.sqrt(mean_squared_error(y_holdout, y_pred_lgb_holdout))
    r2_lgb_holdout = r2_score(y_holdout, y_pred_lgb_holdout)

    print(f"LightGBM Performance ({country_code}):")
    print(f"Test MAE: {mae_lgb:.6f}, RMSE: {rmse_lgb:.6f}, R2: {r2_lgb:.6f}")
    print(f"Hold-out MAE: {mae_lgb_holdout:.6f}, RMSE: {rmse_lgb_holdout:.6f}, R2: {r2_lgb_holdout:.6f}")

    # Feature importance
    plt.figure(figsize=(12, 8))  # Tăng kích thước biểu đồ
    lgb.plot_importance(model_lgb, max_num_features=10)
    plt.title(f'LightGBM Feature Importance ({country_code})')
    plt.subplots_adjust(left=0.35)  # Tăng lề trái để nhãn trục y hiển thị đầy đủ
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{country_code}_lgb_feature_importance.png')
    plt.close()

    # Compare models
    results = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM'],
        'Test MAE': [mae_xgb, mae_lgb],
        'Test RMSE': [rmse_xgb, rmse_lgb],
        'Test R2': [r2_xgb, r2_lgb],
        'Hold-out MAE': [mae_xgb_holdout, mae_lgb_holdout],
        'Hold-out RMSE': [rmse_xgb_holdout, rmse_lgb_holdout],
        'Hold-out R2': [r2_xgb_holdout, r2_lgb_holdout]
    })
    print(f"\nModel Comparison ({country_code}):")
    print(results)

    # Select best model based on hold-out R2
    best_model_name = "XGBoost" if r2_xgb_holdout > r2_lgb_holdout else "LightGBM"
    best_model = model_xgb if best_model_name == "XGBoost" else model_lgb

    print(f"\n✅ Best Model Selected for {country_code}: {best_model_name}")

    # Save best model info
    model_info = {"best_model": best_model_name}
    with open(f'{output_dir}/{country_code}_best_model.json', 'w') as f:
        json.dump(model_info, f)

    # Save model and SHAP values
    print("⏳ Computing SHAP values...")
    start_time = time.time()
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())
    if best_model_name == "XGBoost":
        joblib.dump(model_xgb, f'{output_dir}/{country_code}_model_xgb.joblib')
        explainer = shap.TreeExplainer(model_xgb, X_test_clean, approximate=True)
        shap_values = explainer(X_test_clean)
        joblib.dump(shap_values, f'{output_dir}/{country_code}_shap_values_xgb.joblib')
        shap.summary_plot(shap_values, X_test_clean, show=False)
        plt.savefig(f'{output_dir}/{country_code}_shap_summary.png')
        plt.close()
    else:
        joblib.dump(model_lgb, f'{output_dir}/{country_code}_model_lgb.joblib')
        explainer = shap.TreeExplainer(model_lgb, X_test_clean, approximate=True)
        shap_values = explainer(X_test_clean, check_additivity=False)
        joblib.dump(shap_values, f'{output_dir}/{country_code}_shap_values_lgb.joblib')
        shap.summary_plot(shap_values, X_test_clean, show=False)
        plt.savefig(f'{output_dir}/{country_code}_shap_summary.png')
        plt.close()
    print(f" - SHAP computation time: {time.time() - start_time:.2f} seconds")

    # Save X_test for recommendation
    X_test.to_csv(f'{output_dir}/{country_code}_X_test.csv', index=True)

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    for country in countries:
        train_model(country)

