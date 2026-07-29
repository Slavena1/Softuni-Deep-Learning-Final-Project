"""
modeling.py
Train/test splitting and model pipelines for the Ridge, Lasso, and
Random Forest models used in Sections 4 and 5.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

RANDOM_STATE = 42


def split_raw(df, features, target_col, test_size=0.25, stratify_col=None):
    """
    Split raw (unscaled) features and target, returning train/test row
    indices alongside the split arrays so predictions can be traced back
    to specific words (used in Section 5.5).
    """
    X = df[features].values
    y = df[target_col].values
    idx = df.index.values
    stratify = df[stratify_col].values if stratify_col else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def fit_ridge(X_train, y_train, param_grid, cv):
    """Grid-search Ridge inside a scaling pipeline, fit on training data only."""
    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
    grid = GridSearchCV(pipe, {'ridge__alpha': param_grid['alpha']}, cv=cv, scoring='r2')
    grid.fit(X_train, y_train)
    return grid


def fit_lasso(X_train, y_train, param_grid, cv):
    """Grid-search Lasso inside a scaling pipeline, fit on training data only."""
    pipe = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(max_iter=10000))])
    grid = GridSearchCV(pipe, {'lasso__alpha': param_grid['alpha']}, cv=cv, scoring='r2')
    grid.fit(X_train, y_train)
    return grid


def fit_rf_classifier(X_train, y_train, n_estimators=200, random_state=42):
    """Random Forest classifier inside a scaling pipeline, fit on training data only."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def error_analysis(df_analysis, pipe, X_test, y_test, test_index, word_col='word', top_n=15):
    """
    Identify the worst-predicted words on the held-out test set and
    report which word categories (if available) they cluster in.
    """
    y_pred = pipe.predict(X_test)
    errors = pd.DataFrame({
        'word': df_analysis.loc[test_index, word_col].values,
        'actual': y_test,
        'predicted': y_pred,
        'abs_error': np.abs(y_test - y_pred)
    })
    if 'category' in df_analysis.columns:
        errors['category'] = df_analysis.loc[test_index, 'category'].values

    worst = errors.sort_values('abs_error', ascending=False).head(top_n)

    category_breakdown = None
    if 'category' in errors.columns:
        category_breakdown = (
            errors.groupby('category')['abs_error']
            .agg(['mean', 'count'])
            .sort_values('mean', ascending=False)
        )
    return worst, category_breakdown
