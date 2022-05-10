from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.tree import DecisionTreeClassifier

def create_pipeline(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
def create_pipeline_both_model(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int, f_eng: str, cl: str, max_depth: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    selection_model = RandomForestClassifier(max_depth=13)
    if f_eng == 'SequentialFeatureSelector':
        pipeline_steps.append(
            (
            "feature_selection",
            VarianceThreshold()
            )    
        )
    elif f_eng == 'SelectFromModel':
        pipeline_steps.append(
            (
            "feature_selection",
            SelectFromModel(estimator=selection_model)
            )    
        )    
   
    if cl == 'LogisticRegression':
        
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_C
                ),
            )
        )
    elif cl == 'DecisionTreeClassifier':
        pipeline_steps.append(
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=random_state, max_depth=max_depth
                ),
            )
        )
    return Pipeline(steps=pipeline_steps)