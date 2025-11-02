# tools/inspect_model.py
import numpy as np
import pandas as pd
import joblib

def get_feature_names(preprocessor) -> list:
    """
    Robustly extract output feature names from a ColumnTransformer.
    - For one-hot: use get_feature_names_out with original column names
    - For passthrough numerics: just return the original column names
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue

        # Pipelines (e.g., imputer -> onehot)
        if hasattr(transformer, "named_steps"):
            # Get the last step (usually the encoder)
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                out = last_step.get_feature_names_out(cols)
                feature_names.extend(out)
            else:
                # No name expansion (e.g., only imputer) -> keep original
                feature_names.extend(list(cols))
        else:
            # Plain transformer like "passthrough" or OneHotEncoder itself
            if hasattr(transformer, "get_feature_names_out"):
                out = transformer.get_feature_names_out(cols)
                feature_names.extend(out)
            else:
                # passthrough numerics
                feature_names.extend(list(cols))
    return list(feature_names)

def main():
    pipe = joblib.load("artifacts/model.joblib")
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]

    # Get final output feature names in the SAME order used by the model
    feature_names = get_feature_names(pre)

    # Coefficients (1D array)
    coefs = np.asarray(model.coef_).ravel()

    # Build a tidy frame
    coefdf = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    })
    coefdf["odds_ratio"] = np.exp(coefdf["coef"])

    # Sort views
    top_risk = coefdf.sort_values("coef", ascending=False).head(15)
    top_protective = coefdf.sort_values("coef", ascending=True).head(15)

    # Pretty print
    pd.set_option("display.max_colwidth", 120)
    print("\n==== Top features INCREASING churn risk (positive weights) ====")
    print(top_risk.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n==== Top features DECREASING churn risk (negative weights) ====")
    print(top_protective.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # saving to CSV 
    coefdf.sort_values("coef", ascending=False).to_csv("artifacts/coefficients_desc.csv", index=False)
    print("\nSaved full coefficients to artifacts/coefficients_desc.csv")

    

if __name__ == "__main__":
    main()
