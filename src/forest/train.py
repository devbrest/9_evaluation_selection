from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import v_measure_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from .data import get_dataset
from .pipeline import create_pipeline, create_pipeline_both_model


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
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--max_depth",
    default=13,
    type=int,
    show_default=True,
)
@click.option(
    "--f_eng",
    default="None",
    type=click.Choice(['None','SelectFromModel', 'VarianceThreshold'], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--cl",
    default="LogisticRegression",
    type=click.Choice(['LogisticRegression', 'DecisionTreeClassifier'], case_sensitive=False),
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    max_depth: int,
    f_eng: str,
    cl: str
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        """
        cv = KFold(n_splits=5, random_state=random_state, shuffle=True)
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        scores = cross_val_score(pipeline, features_train, target_train, scoring='accuracy',
                         cv=cv)
        accuracy = scores.mean()
        scores = cross_val_score(pipeline, features_train, target_train, scoring='r2',
                         cv=cv)
        r2_score_val = scores.mean()
        scores = cross_val_score(pipeline, features_train, target_train, scoring='v_measure_score',
                         cv=cv)
        v_measure_score_val = scores.mean()

        #pipeline.fit(features_train, target_train)
        #predict_val = pipeline.predict(features_val)
        #accuracy = accuracy_score(target_val, predict_val)
        #r2_score_val = r2_score(target_val, predict_val)
        #v_measure_score_val = v_measure_score(target_val, predict_val)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("r2_score", r2_score_val)
        mlflow.log_metric("v_measure_score", v_measure_score_val)
        mlflow.sklearn.log_model(pipeline, "model")
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"R2_score: {r2_score_val}.")
        click.echo(f"V_measure_score: {v_measure_score_val}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")""";
        pipeline = create_pipeline_both_model(use_scaler, max_iter, logreg_c, random_state, f_eng, cl,max_depth)
        pipeline.fit(features_train, target_train)
        predict_val = pipeline.predict(features_val)
        accuracy = accuracy_score(target_val, predict_val)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("feature_selection", f_eng)
        mlflow.log_param("classifier", cl)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")

