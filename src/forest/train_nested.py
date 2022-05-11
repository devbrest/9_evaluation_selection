from pathlib import Path
from joblib import dump

import numpy as np
import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import v_measure_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from .data import get_dataset
from .pipeline import create_pipeline_nested


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-nested",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max_depth",
    default=13,
    type=int,
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=100,
    type=int,
    show_default=True,
)

@click.option(
    "--max_features",
    default="None",
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=click.Choice(['gini', 'entropy'], case_sensitive=False),
    show_default=True,
)
def train_nested(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,    
    use_nested: bool,
    max_depth: int,
    n_estimators: int,
    max_features: int,
    criterion: str
    
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio
    )
    with mlflow.start_run():
        if use_nested == True:
            
            
            param_grid = {
                'max_depth': [10, 20, 40, None],
                'n_estimators': np.arange(1, 201, step=50),
                'max_features':["auto", "sqrt", "log2"],
                'criterion': ['gini','entropy']
            }
            #cv_outer = KFold(n_splits=3, random_state=None, shuffle=True)
            cv_outer = KFold(n_splits=2)
            outer_results = list()
            for train_ix, test_ix in cv_outer.split(features_train):
                
                # split data
                #print(train_ix)
                X_train, X_test = features_train.iloc[train_ix], features_train.iloc[test_ix]
                y_train, y_test = target_train.iloc[train_ix], target_train.iloc[test_ix]
                #X_train = features_train.iloc[train_ix]
                #y_train = target_train.iloc[train_ix]
            
                
            # configure the cross-validation procedure
                cv_inner = KFold(n_splits=2)
            # define the model
                model = RandomForestClassifier() #create_pipeline_nested()
            # define search
                search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv_inner, refit=True)
                # execute search
                result = search.fit(X_train, y_train)
                # get the best performing model fit on the whole training set
                best_model = result.best_estimator_
                # evaluate model on the hold out dataset
                yhat = best_model.predict(X_test)
                # evaluate the model
                acc = accuracy_score(y_test, yhat)
                # store the result
                outer_results.append(acc)
                mlflow.log_param("max_depth", best_model.max_depth)
                mlflow.log_param("n_estimators", best_model.n_estimators)
                mlflow.log_param("max_features", best_model.max_features)
                mlflow.log_param("criterion", best_model.criterion)
                
                mlflow.log_metric("accuracy", acc)
        else:
            
                

            pipeline = create_pipeline_nested(max_depth=max_depth,n_estimators=n_estimators,max_features=max_features,criterion=criterion)

            pipeline.fit(features_train, target_train)
            predict_val = pipeline.predict(features_val)
            accuracy = accuracy_score(target_val, predict_val)
            r2_score_val = r2_score(target_val, predict_val)
            v_measure_score_val = v_measure_score(target_val, predict_val)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("criterion", criterion)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("r2_score", r2_score_val)
            mlflow.log_metric("v_measure_score", v_measure_score_val)
            mlflow.sklearn.log_model(pipeline, "model")
            click.echo(f"Accuracy: {accuracy}.")
            click.echo(f"R2_score: {r2_score_val}.")
            click.echo(f"V_measure_score: {v_measure_score_val}.")
            dump(pipeline, save_model_path)
            click.echo(f"Model is saved to {save_model_path}.")
            

