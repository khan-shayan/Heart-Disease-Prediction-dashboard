import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import importlib
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# ------------------ Page + Styling ------------------
st.set_page_config(page_title="Classification Model Dashboard", page_icon="🫀", layout="wide")
st.title(" ")
st.title("🫀 Heart Disease Prediction Model")

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3, h4 { color: #e6e6e6 !important; }
    .stMarkdown, .stDataFrame { color: #ddd; }
    .metric-card { background: #161a23; padding: 1rem; border-radius: 0.5rem; border: 1px solid #2a2f3a; }
    </style>
    """,
    unsafe_allow_html=True,
)

feature_units = {
    "age": "years",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "thalach": "bpm",
    "oldpeak": "mm",
    # categorical/binary features can have empty string or description
    "sex": "",
    "cp": "type",
    "fbs": "",
    "restecg": "",
    "exang": "",
    "slope": "",
    "ca": "",
    "thal": "",
    "target": "",
}

# ------------------ Session State ------------------
ss = st.session_state
ss.setdefault("trained_models", {})          # model_name -> fitted model
ss.setdefault("feature_names", None)         # list of feature names
ss.setdefault("label_encoder", None)         # fitted LabelEncoder for target
ss.setdefault("best_params", {})             # model_name -> best params (from GridSearch)
ss.setdefault("results", {})                 # model_name -> classification_report dict
ss.setdefault("cmats", {})                   # model_name -> confusion matrix (np.array)
ss.setdefault("splits", None)                # (X_train, X_test, y_train, y_test)
ss.setdefault("df_meta", None)               # metadata about uploaded df to detect change
ss.setdefault("shap_cache", {})              # model_name -> {"values": array/list, "X_sample": DataFrame}
ss.setdefault("tuned", False)
ss.setdefault("trained", False)

if "ensemble_models" not in ss:
    ss["ensemble_models"] = {}


def reset_everything():
    keys = ["trained_models", "feature_names", "label_encoder", "best_params",
            "results", "cmats", "splits", "shap_cache", "tuned", "trained"]
    for k in keys:
        ss[k] = {} if k in ["trained_models", "best_params", "results", "cmats", "shap_cache"] else None
    ss["tuned"] = False
    ss["trained"] = False

# ------------------ Models ------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models_available = {
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "LightGBM": LGBMClassifier,
    "XGBoost": XGBClassifier,
}

# ------------------ Sidebar ------------------
st.sidebar.header("Upload & Model Selection")
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

selected_models = st.sidebar.multiselect(
    "Select Models to Train",
    list(models_available.keys()),
    default=["Decision Tree", "Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"],
    help="Choose one or more models which you want to train on the dataset."
)

test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.25, 0.05, help="Select data on which you will perform testing.")
random_state = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1, help="A random seed (or seed state, or just seed) is a number (or vector) used to initialize a pseudorandom number generator.")

# Reset button
if st.sidebar.button("🔁 Reset tuning & training (clear cache)"):
    reset_everything()
    st.sidebar.success("Cleared. Re-run or change tabs to refresh.")

# ------------------ Param grids (yours, intact with minor fixes) ------------------
param_grids = {
    "Decision Tree": {
        "max_depth": [3, 5, 7],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2],
    },
    "Random Forest": {
        "n_estimators": [30, 50, 100],
        "max_depth": [2, 5],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [2, 3],
        "bootstrap": [True, False],
    },
    "Gradient Boosting": {
        "n_estimators": [30, 50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "min_samples_split": [3, 4],
        "min_samples_leaf": [1, 2],
        "subsample": [0.6, 0.8],
        "max_features": ["sqrt", "log2", None],
    },
    "LightGBM": {
        "n_estimators": [30, 50, 100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 50],
        "max_depth": [-1, 5],
        "min_child_samples": [30, 40],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "reg_alpha": [0.1, 0.5],
        "reg_lambda": [0.1, 0.5],
    },
    "XGBoost": {
        "n_estimators": [30],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.6, 0.8],
        "min_child_weight": [3, 5],
        "gamma": [0.0, 0.1],              # <- allow 0.0 (your slider default)
        "colsample_bytree": [0.6, 0.8],
        "reg_alpha": [0.0, 0.5],
        "reg_lambda": [0.0, 0.5],
    },
}

# ------------------ Helper: detect new dataset ------------------
def dataset_changed(df, file_name):
    meta = {"rows": df.shape[0], "cols": df.shape[1], "name": file_name}
    changed = (ss["df_meta"] != meta)
    ss["df_meta"] = meta
    return changed

# ------------------ UI Tabs ------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # If a new dataset is uploaded, clear caches
    if dataset_changed(df, uploaded_file.name):
        reset_everything()

    preprocessing_tab, explore_tab, train_tab, ensemble_tab, results_tab, explain_tab, predict_tab = st.tabs(
        [ "Preprocessing", "Explore Data", "Train & Tune", "Ensemble", "Results", "Explanation", "Live Prediction"]
    )

# ------------------ Preprocessing ------------------
    with preprocessing_tab:
        st.subheader(
            "Data Preprocessing",
            help="Clean, transform, and prepare your raw data before training the model. "
                "This step handles missing values, encodes categorical variables, and scales features to improve model performance and accuracy."
        )

        # Store raw data in session state
        st.session_state["raw_data"] = df

        # Set default target column
        default_target = "target"  # change this to the column you want
        target_col = st.selectbox(
            "Select target column", 
            df.columns, 
            index=df.columns.get_loc(default_target) if default_target in df.columns else 0
            )

        # ---------- Preprocessing options ----------
        st.write("### Select Preprocessing Steps")
        remove_nulls = st.checkbox("Remove Null Values")
        remove_duplicates = st.checkbox("Remove Duplicates")
        fill_missing_values = st.checkbox("Fill Missing Values")
        encode_categorical = st.checkbox("Encode Categorical Columns")
        scale_features = st.checkbox("Scale Numeric Features")
        # You can add more options here

        if st.button("Run Preprocessing"):
            df_processed = df.copy()

            # ---------- Apply selected preprocessing ----------
            if remove_nulls:
                df_processed = df_processed.dropna()
                st.success("✅ Null values removed")

            if remove_duplicates:
                df_processed = df_processed.drop_duplicates()
                st.success("✅ Duplicates removed")

            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]

            if fill_missing_values:
                # Fill numeric NaNs with median
                num_cols = X.select_dtypes(include=["number"]).columns
                X[num_cols] = X[num_cols].fillna(X[num_cols].median())

                # Fill categorical NaNs with "Unknown"
                cat_cols = X.select_dtypes(include=["object"]).columns
                X[cat_cols] = X[cat_cols].fillna("Unknown")

                st.success("✅ Missing values filled (numeric → median, categorical → 'Unknown')")

            if encode_categorical:
                cat_cols = X.select_dtypes(include=["object"]).columns
                if len(cat_cols) > 0:
                    st.write("Encoding categorical columns:", list(cat_cols))
                    X = pd.get_dummies(X, columns=cat_cols)

                # Encode target if categorical
                if y.dtype == "object":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state["label_encoder"] = le
                else:
                    st.session_state["label_encoder"] = None
                st.success("✅ Categorically encoded")
            else:
                st.session_state["label_encoder"] = None

            if scale_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.success("✅ Features scaled")
            else:
                X_scaled = X.values
                scaler = None

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Save to session state
            st.session_state["processed_data"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "feature_names": X.columns.tolist(),
                "scaler": scaler,
                "target_col": target_col
            }

            st.success("✅ Preprocessing complete! Data stored in session state.")

            # Preview processed dataset
            st.subheader("Processed dataset preview")
            st.write(pd.DataFrame(X_scaled, columns=X.columns))

# ------------------ Explore Data ------------------
    with explore_tab:
        # ------------------ Data Visualizations ------------------
        continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        st.subheader("Distribution of Continuous Variables")
        df_continuous = df[continuous_features]
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        for i, col in enumerate(df_continuous.columns):
            x, y = i // 3, i % 3
            values, bin_edges = np.histogram(df_continuous[col],
                                            range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))

            graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
                                edgecolor='none', color='red', alpha=0.6, line_kws={'lw': 3})
            ax[x, y].set_xlabel(col, fontsize=15)
            ax[x, y].set_ylabel('Count', fontsize=12)
            ax[x, y].set_xticks(np.round(bin_edges, 1))
            ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
            ax[x, y].grid(color='cyan')

            for j, p in enumerate(graph.patches):
                ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                                ha='center', fontsize=10, fontweight="bold")

            textstr = '\n'.join((
                f"mu = {df_continuous[col].mean():.2f}",
                f"sigma = {df_continuous[col].std():.2f}"
                ))
            ax[x, y].text(0.8, 1, textstr, transform=ax[x, y].transAxes, verticalalignment='top',
                        color='white', bbox=dict(boxstyle='round', facecolor="#1201faff", edgecolor='white', pad=0.5))

        ax[1, 2].axis('off')
        plt.suptitle('Distribution of Continuous Variables', fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        st.pyplot(fig)
        plt.close(fig)

        # ------------------ Categorical Features Distribution ------------------
        st.subheader("Distribution of Categorical Variables")

        categorical_features = df.columns.difference(continuous_features)
        df_categorical = df[categorical_features]

        n_features = len(categorical_features)
        ncols = 5
        nrows = (n_features + ncols - 1) // ncols

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
        ax = ax.flatten()

        for i, col in enumerate(categorical_features):
            value_counts = df[col].value_counts(normalize=True).mul(100).sort_values()
            value_counts.plot(kind='bar', ax=ax[i], width=0.7, color='red', alpha=0.6)

            for index, value in enumerate(value_counts):
                ax[i].text(index, value + 1, str(round(value, 1)) + '%',
                        fontsize=12, weight='bold', ha='center', color='blue')

            ax[i].set_ylim([0, 105])
            ax[i].set_ylabel('Frequency Percentage', fontsize=12)
            ax[i].set_title(f'{col}', fontsize=16)
            ax[i].set_xticklabels(ax[i].get_xticklabels(), ha='right')

        for j in range(i + 1, len(ax)):
            ax[j].axis('off')

        plt.suptitle('Distribution of Categorical Variables', fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        st.pyplot(fig)
        plt.close(fig)

        # ------------------ Continuous vs Target ------------------
        st.subheader("Continuous Features vs Target")

        sns.set_palette(["#4E3BFC", "#FC4747"])  # Blue vs Red

        n_features = len(continuous_features)
        fig, ax = plt.subplots(
            n_features, 2,
            figsize=(15, 3 * n_features),
            gridspec_kw={'width_ratios': [1, 4]}
        )

        if n_features == 1:
            ax = np.array([ax])

        for i, col in enumerate(continuous_features):
            # Barplot
            graph = sns.barplot(data=df, x="target", y=col, ax=ax[i, 0],
                                palette=["#4E3BFC", "#FC4747"])

            # KDE plots with distinct colors
            sns.kdeplot(data=df[df["target"] == 0], x=col, fill=True, linewidth=2,
                        ax=ax[i, 1], label='0', color="#4E3BFC")   # Blue
            sns.kdeplot(data=df[df["target"] == 1], x=col, fill=True, linewidth=2,
                        ax=ax[i, 1], label='1', color="#FC4747")   # Red

            ax[i, 1].set_yticks([])
            ax[i, 1].legend(title='Heart Disease', loc='upper right')

            for cont in graph.containers:
                graph.bar_label(cont, fmt='         %.3g')

        plt.suptitle('Continuous Features vs Target Distribution', fontsize=22)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ------------------ Categorical vs Target ------------------
        st.subheader("Categorical Features vs Target")

        n_features = len(categorical_features)
        ncols = 4
        nrows = (n_features + ncols - 1) // ncols

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))

        if isinstance(ax, np.ndarray):
            ax = ax.flatten()
        else:
            ax = np.array([ax])

        for i, col in enumerate(categorical_features):
            cross_tab = pd.crosstab(index=df[col], columns=df['target'])
            cross_tab_prop = pd.crosstab(index=df[col], columns=df['target'], normalize='index')
            cmp = ListedColormap(['#4E3BFC', '#FC4747'])  # blue vs Red

            cross_tab_prop.plot(
                kind='bar', ax=ax[i], stacked=True, width=0.8, colormap=cmp,
                legend=False, ylabel='Proportion', sharey=True
            )

            # Annotate bars
            for idx, val in enumerate(cross_tab.index.values):
                for (proportion, count, y_location) in zip(
                    cross_tab_prop.loc[val],
                    cross_tab.loc[val],
                    cross_tab_prop.loc[val].cumsum()
                ):
                    ax[i].text(
                        x=idx - 0.3,
                        y=(y_location - proportion) + (proportion / 2) - 0.03,
                        s=f'    {count}\n({np.round(proportion * 100, 1)}%)',
                        color="black", fontsize=9, fontweight="bold"
                    )

            ax[i].legend(title='Target', loc='upper right', fontsize=8)
            ax[i].set_ylim([0, 1.12])
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0)

        for j in range(i + 1, len(ax)):
            fig.delaxes(ax[j])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ------------------Feature Correlation ------------------
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(df.corr(), annot=True, cmap='inferno', linewidths=0.5, ax=ax)
        plt.title('Feature Correlation Matrix')
        st.pyplot(fig)
        plt.close(fig)

    # ------------------ Train & Tune ------------------
    with train_tab:
        # Target column (default last column)
        target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns) - 1)

        # Prepare X, y and split (store once)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)

        # Save LE for all other tabs
        ss["label_encoder"] = label_encoder

        # Build/refresh splits if not present or if test_size/random_state changed
        need_new_split = (
            ss["splits"] is None
            or ss.get("split_meta") != (test_size, random_state, tuple(X.columns))
        )
        if need_new_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state
            )
            ss["splits"] = (X_train, X_test, y_train, y_test)
            ss["feature_names"] = X.columns.tolist()
            ss["split_meta"] = (test_size, random_state, tuple(X.columns))

        X_train, X_test, y_train, y_test = ss["splits"]

        # ---------- 1) Auto-tune (run once unless Reset) ----------
        st.subheader("1) Auto-Tuning (GridSearchCV)", help="Automatically finds the best machine learning model parameters by testing all possible combinations in a 'grid'. This ensures optimal performance but can be computationally expensive.")
        colA, colB = st.columns([1, 3])
        with colA:
            tune_clicked = st.button("🔍 Run Auto-Tuning now")
        with colB:
            if ss["tuned"]:
                st.success("Best params already computed and cached ✅")
            else:
                st.info("Click 'Run Auto-Tuning now' to compute best params.")

        if tune_clicked:
            with st.spinner("Running grid search..."):
                best_params = {}
                for model_name in selected_models:
                    ModelClass = models_available[model_name]
                    # All models we use accept random_state; for safety, only pass if supported
                    try:
                        base = ModelClass(random_state=random_state)
                    except TypeError:
                        base = ModelClass()
                    grid = GridSearchCV(
                        base,
                        param_grids[model_name],
                        cv=3,
                        scoring="f1_weighted",
                        n_jobs=-1,
                    )
                    grid.fit(X_train, y_train)
                    best_params[model_name] = grid.best_params_
                    st.write(f"Best params for **{model_name}**: ", best_params[model_name])

                ss["best_params"] = best_params
                ss["tuned"] = True
                st.success("Auto-tuning complete and cached ✅")

        # ---------- 2) Hyperparams UI (defaults from best params if available) ----------
        st.subheader("2) Set Hyperparameters (defaults from Auto-Tuning)", help="Manually adjust the model parameters. The input fields are pre-filled with the optimal values found by the Auto-Tuning process, but you can customize them here.")
        def get_model_params(model_name):
            params = {}
            defaults = ss["best_params"].get(model_name, {})
            # Utility: map None safely for selectbox
            def _index_or_default(opts, val, fallback):
                val = "None" if val is None else val
                opts_conv = ["None" if o is None else o for o in opts]
                return opts_conv.index(val) if val in opts_conv else opts_conv.index(fallback)

            if model_name == "Decision Tree":
                params["max_depth"] = st.sidebar.slider(
                    "DecisionTree: max_depth", 
                    1, 20, int(defaults.get("max_depth", 3)),
                    step=1, 
                    key=f"{model_name}_max_depth",
                    help="Maximum depth of the tree. Deeper trees can learn more complex patterns but may overfit. Shallower trees are simpler and more general."
                )
                params["min_samples_split"] = st.sidebar.slider(
                    "DecisionTree: min_samples_split", 
                    2, 20, int(defaults.get("min_samples_split", 2)),
                    step=1, 
                    key=f"{model_name}_min_samples_split",
                    help="Minimum number of samples required to split an internal node. Higher values prevent overfitting by limiting splits on small groups."
                )
                params["min_samples_leaf"] = st.sidebar.slider(
                    "DecisionTree: min_samples_leaf", 
                    1, 20, int(defaults.get("min_samples_leaf", 1)),
                    step=1, 
                    key=f"{model_name}_min_samples_leaf",
                    help="Minimum number of samples required to be at a leaf node. Higher values regularize the model by preventing leaves with few samples."
                )
                params["criterion"] = st.sidebar.selectbox(
                    "DecisionTree: criterion", 
                    ["gini", "entropy"],
                    index=_index_or_default(["gini", "entropy"], defaults.get("criterion", "entropy"), "entropy"),
                    key=f"{model_name}_criterion",
                    help="Function to measure the quality of a split. 'Gini' measures impurity, 'Entropy' measures information gain. Both typically yield similar results."
                )

            elif model_name == "Random Forest":
                params["n_estimators"] = st.sidebar.slider(
                    "RandomForest: n_estimators", 
                    10, 1000, int(defaults.get("n_estimators", 100)),
                    step=10, 
                    key=f"{model_name}_n_estimators",
                    help="Number of trees in the forest. More trees generally improve performance but increase computation time. Typically 100-500 is sufficient."
                )
                params["max_depth"] = st.sidebar.slider(
                    "RandomForest: max_depth", 
                    1, 50, int(defaults.get("max_depth", 5)),
                    step=1, 
                    key=f"{model_name}_max_depth",
                    help="Maximum depth of each tree. Deeper trees can capture more complex patterns but may lead to overfitting. Shallower trees create simpler models."
                )
                params["min_samples_split"] = st.sidebar.slider(
                    "RandomForest: min_samples_split", 
                    2, 20, int(defaults.get("min_samples_split", 2)),
                    step=1, 
                    key=f"{model_name}_min_samples_split",
                    help="Minimum number of samples required to split an internal node. Higher values prevent overfitting by limiting splits on small sample groups."
                )
                params["min_samples_leaf"] = st.sidebar.slider(
                    "RandomForest: min_samples_leaf", 
                    1, 20, int(defaults.get("min_samples_leaf", 2)),
                    step=1, 
                    key=f"{model_name}_min_samples_leaf",
                    help="Minimum number of samples required to be at a leaf node. Higher values create more generalized models by preventing small leaf nodes."
                )
                params["bootstrap"] = st.sidebar.selectbox(
                    "RandomForest: bootstrap", 
                    [True, False],
                    index=[True, False].index(bool(defaults.get("bootstrap", True))),
                    key=f"{model_name}_bootstrap",
                    help="Whether to use bootstrap samples when building trees. If False, the whole dataset is used. Bootstrap=True typically provides better generalization."
                )
                params["criterion"] = st.sidebar.selectbox(
                    "RandomForest: criterion", 
                    ["gini", "entropy"],
                    index=_index_or_default(["gini", "entropy"], defaults.get("criterion", "gini"), "gini"),
                    key=f"{model_name}_criterion",
                    help="Function to measure split quality. 'Gini' measures impurity (faster), 'Entropy' measures information gain. Both usually yield similar results."
                )

            elif model_name == "Gradient Boosting":
                params["n_estimators"] = st.sidebar.slider(
                    "GBM: n_estimators", 
                    10, 1000, int(defaults.get("n_estimators", 100)),
                    step=10, 
                    key=f"{model_name}_n_estimators",
                    help="Number of boosting stages (trees) to build. More estimators improve performance but increase risk of overfitting and computation time."
                )
                params["learning_rate"] = st.sidebar.slider(
                    "GBM: learning_rate", 
                    0.001, 1.0, float(defaults.get("learning_rate", 0.1)),
                    step=0.01, 
                    key=f"{model_name}_learning_rate",
                    help="Shrinks the contribution of each tree. Lower values require more trees but often lead to better generalization. Trade-off with n_estimators."
                )
                params["max_depth"] = st.sidebar.slider(
                    "GBM: max_depth", 
                    1, 20, int(defaults.get("max_depth", 3)),
                    step=1, 
                    key=f"{model_name}_max_depth",
                    help="Maximum depth of individual trees. Controls model complexity. Shallower trees are more conservative and prevent overfitting."
                )
                params["min_samples_split"] = st.sidebar.slider(
                    "GBM: min_samples_split", 
                    2, 20, int(defaults.get("min_samples_split", 3)),
                    step=1, 
                    key=f"{model_name}_min_samples_split",
                    help="Minimum samples required to split an internal node. Higher values regularize the model by preventing splits on small groups."
                )
                params["min_samples_leaf"] = st.sidebar.slider(
                    "GBM: min_samples_leaf", 
                    1, 20, int(defaults.get("min_samples_leaf", 2)),
                    step=1, 
                    key=f"{model_name}_min_samples_leaf",
                    help="Minimum samples required at leaf nodes. Higher values create smoother predictions and prevent overfitting."
                )
                params["subsample"] = st.sidebar.slider(
                    "GBM: subsample", 
                    0.1, 1.0, float(defaults.get("subsample", 0.6)),
                    step=0.1, 
                    key=f"{model_name}_subsample",
                    help="Fraction of samples used for fitting individual trees. Values < 1.0 introduce randomness that can prevent overfitting (stochastic gradient boosting)."
                )
                params["max_features"] = st.sidebar.selectbox(
                    "GBM: max_features", 
                    ["sqrt", "log2", None],
                    index=_index_or_default(["sqrt", "log2", None], defaults.get("max_features", "sqrt"), "sqrt"),
                    key=f"{model_name}_max_features",
                    help="Number of features to consider for best split. 'sqrt' and 'log2' use fewer features, adding diversity. None uses all features."
                )

            elif model_name == "LightGBM":
                params["n_estimators"] = st.sidebar.slider(
                    "LGBM: n_estimators", 
                    10, 1000, int(defaults.get("n_estimators", 100)),
                    step=10, 
                    key=f"{model_name}_n_estimators",
                    help="Number of boosting iterations (trees). More trees can improve performance but may lead to overfitting. Balance with learning_rate."
                )
                params["learning_rate"] = st.sidebar.slider(
                    "LGBM: learning_rate", 
                    0.001, 1.0, float(defaults.get("learning_rate", 0.1)),
                    step=0.01, 
                    key=f"{model_name}_learning_rate",
                    help="Shrinkage rate for tree contributions. Lower values make the model more robust but require more trees. Typical range: 0.01-0.3."
                )
                params["num_leaves"] = st.sidebar.slider(
                    "LGBM: num_leaves", 
                    2, 512, int(defaults.get("num_leaves", 31)),
                    step=1, 
                    key=f"{model_name}_num_leaves",
                    help="Maximum number of leaves in one tree. The main parameter to control complexity. Higher values may overfit, lower may underfit."
                )
                params["max_depth"] = st.sidebar.slider(
                    "LGBM: max_depth", 
                    -1, 50, int(defaults.get("max_depth", -1)),
                    step=1, 
                    key=f"{model_name}_max_depth",
                    help="Maximum tree depth. -1 means no limit. Shallower trees are faster and prevent overfitting. Use with num_leaves for complexity control."
                )
                params["min_child_samples"] = st.sidebar.slider(
                    "LGBM: min_child_samples", 
                    1, 200, int(defaults.get("min_child_samples", 40)),
                    step=1, 
                    key=f"{model_name}_min_child_samples",
                    help="Minimum number of data points in a leaf. Higher values prevent overfitting by limiting growth on small sample groups."
                )
                params["subsample"] = st.sidebar.slider(
                    "LGBM: subsample", 
                    0.1, 1.0, float(defaults.get("subsample", 0.6)),
                    step=0.1, 
                    key=f"{model_name}_subsample",
                    help="Subsample ratio of the training instances. Lower values add randomness and prevent overfitting (stochastic gradient boosting)."
                )
                params["colsample_bytree"] = st.sidebar.slider(
                    "LGBM: colsample_bytree", 
                    0.1, 1.0, float(defaults.get("colsample_bytree", 0.6)),
                    step=0.1, 
                    key=f"{model_name}_colsample_bytree",
                    help="Subsample ratio of columns when constructing each tree. Lower values add feature randomness and prevent overfitting."
                )
                params["reg_alpha"] = st.sidebar.slider(
                    "LGBM: reg_alpha", 
                    0.0, 5.0, float(defaults.get("reg_alpha", 0.5)),
                    step=0.1, 
                    key=f"{model_name}_reg_alpha",
                    help="L1 regularization term on weights. Higher values make the model more conservative by pushing feature weights to zero."
                )
                params["reg_lambda"] = st.sidebar.slider(
                    "LGBM: reg_lambda", 
                    0.0, 5.0, float(defaults.get("reg_lambda", 0.5)),
                    step=0.1, 
                    key=f"{model_name}_reg_lambda",
                    help="L2 regularization term on weights. Higher values make the model more conservative by smoothing feature weights."
                )

            elif model_name == "XGBoost":
                params["n_estimators"] = st.sidebar.slider(
                    "XGB: n_estimators", 
                    10, 1000, int(defaults.get("n_estimators", 30)),
                    step=10, 
                    key=f"{model_name}_n_estimators",
                    help="Number of boosting rounds (trees). More trees can improve performance but increase training time and risk of overfitting."
                )
                params["learning_rate"] = st.sidebar.slider(
                    "XGB: learning_rate", 
                    0.001, 1.0, float(defaults.get("learning_rate", 0.1)),
                    step=0.01, 
                    key=f"{model_name}_learning_rate",
                    help="Step size shrinkage used to prevent overfitting. Lower values require more trees but often yield better generalization. Typical range: 0.01-0.3."
                )
                params["max_depth"] = st.sidebar.slider(
                    "XGB: max_depth", 
                    1, 20, int(defaults.get("max_depth", 5)),
                    step=1, 
                    key=f"{model_name}_max_depth",
                    help="Maximum depth of a tree. Increasing depth makes the model more complex and able to learn more patterns, but may cause overfitting."
                )
                params["subsample"] = st.sidebar.slider(
                    "XGB: subsample", 
                    0.1, 1.0, float(defaults.get("subsample", 0.8)),
                    step=0.1, 
                    key=f"{model_name}_subsample",
                    help="Subsample ratio of the training instances. Lower values prevent overfitting but might make the model too conservative."
                )
                params["min_child_weight"] = st.sidebar.slider(
                    "XGB: min_child_weight", 
                    1, 20, int(defaults.get("min_child_weight", 3)),
                    step=1, 
                    key=f"{model_name}_min_child_weight",
                    help="Minimum sum of instance weight (hessian) needed in a child. Higher values prevent overfitting by limiting tree growth."
                )
                params["gamma"] = st.sidebar.slider(
                    "XGB: gamma", 
                    0.0, 1.0, float(defaults.get("gamma", 0.0)),
                    step=0.1, 
                    key=f"{model_name}_gamma",
                    help="Minimum loss reduction required to make a further partition on a leaf node. Higher values make the algorithm more conservative."
                )
                params["colsample_bytree"] = st.sidebar.slider(
                    "XGB: colsample_bytree", 
                    0.1, 1.0, float(defaults.get("colsample_bytree", 0.6)),
                    step=0.1, 
                    key=f"{model_name}_colsample_bytree",
                    help="Subsample ratio of columns when constructing each tree. Lower values add randomness and help prevent overfitting."
                )
                params["reg_alpha"] = st.sidebar.slider(
                    "XGB: reg_alpha", 
                    0.0, 5.0, float(defaults.get("reg_alpha", 0.0)),
                    step=0.1, 
                    key=f"{model_name}_reg_alpha",
                    help="L1 regularization term on weights. Increasing this value makes the model more conservative by sparsifying feature weights."
                )
                params["reg_lambda"] = st.sidebar.slider(
                    "XGB: reg_lambda", 
                    0.0, 5.0, float(defaults.get("reg_lambda", 0.0)),
                    step=0.1, 
                    key=f"{model_name}_reg_lambda",
                    help="L2 regularization term on weights. Increasing this value makes the model more conservative by smoothing feature weights."
                )
            return params

        # Collect params for currently selected models
        current_params = {m: get_model_params(m) for m in selected_models}

        # ---------- 3) Train (fit once; cached in session) ----------
        st.write("### 3) Train Models")
        train_clicked = st.button("🏋️ Train with current hyperparameters")
        if train_clicked:
            with st.spinner("Training models..."):
                results = {}
                cmats = {}
                trained_models = {}

                for model_name in selected_models:
                    ModelClass = models_available[model_name]

                    # Build model with user-chosen params
                    try:
                        model = ModelClass(**current_params[model_name], random_state=random_state)
                    except TypeError:
                        # Some wrappers may not accept random_state
                        model = ModelClass(**current_params[model_name])

                    # Configure eval_set where supported
                    eval_metric = "logloss" if num_classes == 2 else "mlogloss"
                    fit_kwargs = {}
                    if model_name in ["XGBoost", "LightGBM"]:
                        try:
                            fit_kwargs = {"eval_set": [(X_train, y_train), (X_test, y_test)]}
                            if model_name == "XGBoost":
                                try:
                                    model.set_params(eval_metric=eval_metric)
                                except Exception:
                                    pass
                        except Exception:
                            fit_kwargs = {}

                    model.fit(X_train, y_train, **fit_kwargs)
                    trained_models[model_name] = model

                    # Predictions for stored metrics
                    y_pred = model.predict(X_test)
                    le = ss["label_encoder"]
                    y_test_labels = le.inverse_transform(y_test)
                    y_pred_labels = le.inverse_transform(y_pred)

                    report = classification_report(
                        y_test_labels, y_pred_labels, output_dict=True
                    )
                    results[model_name] = report

                    cm = confusion_matrix(
                        y_test_labels, y_pred_labels, labels=le.classes_
                    )
                    cmats[model_name] = cm

                # Persist everything needed for other tabs
                ss["trained_models"] = trained_models
                ss["results"] = results
                ss["cmats"] = cmats
                ss["feature_names"] = X.columns.tolist()
                ss["trained"] = True
                ss["shap_cache"] = {}  # clear SHAP cache as models changed
                st.success("Training complete and cached ✅")

        if ss["trained"]:
            st.success("Using cached trained models ✅")
        elif not ss["tuned"]:
            st.info("Tip: run Auto-Tuning first to pre-fill the sliders with best params.")

    # ------------------ Ensembles ------------------
    with ensemble_tab:
        st.subheader("Ensemble Methods")
        if not ss.get("trained", False) or "trained_models" not in ss:
            st.info("⚠️ Please train models first in the Train & Tune tab before applying ensembles.")
        else:
            selected_ensemble = st.selectbox(
                "Select Ensemble Method",
                ["None", "Averaging", "Stacking"]
            )

            if st.button("Apply Ensemble"):
                if selected_ensemble == "None":
                    st.info("No ensemble selected. Nothing to do.")
                elif not selected_models or len(selected_models) < 2:
                    st.warning("⚠️ Please select at least **two** models in the sidebar to build an ensemble.")
                else:
                    try:
                        from sklearn.base import clone
                        estimators = []
                        for name in selected_models:
                            if name in ss["trained_models"]:
                                estimators.append((name, clone(ss["trained_models"][name])))
                            else:
                                st.warning(f"⚠️ {name} was not trained and will be skipped.")

                        X_train = ss["processed_data"]["X_train"]
                        y_train = ss["processed_data"]["y_train"]
                        X_test = ss["processed_data"].get("X_test", None)
                        y_test = ss["processed_data"].get("y_test", None)

                        # Build ensemble
                        if selected_ensemble == "Averaging":
                            from sklearn.ensemble import VotingClassifier
                            ensemble = VotingClassifier(estimators=estimators, voting="soft")
                            ensemble.fit(X_train, y_train)
                            st.success(f"✅ Averaging ensemble created with: {', '.join(selected_models)}")

                        elif selected_ensemble == "Stacking":
                            from sklearn.ensemble import StackingClassifier
                            from sklearn.linear_model import LogisticRegression
                            ensemble = StackingClassifier(
                                estimators=estimators,
                                final_estimator=LogisticRegression(max_iter=1000),
                                cv=5
                            )
                            ensemble.fit(X_train, y_train)
                            st.success(f"✅ Stacking ensemble created with: {', '.join(selected_models)}")

                        # Save in session state
                        key = f"{selected_ensemble}::" + "_".join(selected_models)
                        if "ensemble_models" not in ss:
                            ss["ensemble_models"] = {}
                        ss["ensemble_models"][key] = ensemble

                        # Evaluate
                        if X_test is not None and y_test is not None:
                            from sklearn.metrics import classification_report, confusion_matrix
                            y_pred = ensemble.predict(X_test)
                            report = classification_report(y_test, y_pred, output_dict=True)
                            cm = confusion_matrix(y_test, y_pred)

                            if "ensemble_results" not in ss:
                                ss["ensemble_results"] = {}
                            ss["ensemble_results"][key] = {"report": report, "confusion_matrix": cm}

                            st.write("### Ensemble classification report")
                            st.dataframe(pd.DataFrame(report).transpose())  # cleaner table format

                            st.write("### Confusion matrix")
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        import traceback
                        st.text(traceback.format_exc())

    # ------------------ Results ------------------
    with results_tab:
        if not ss.get("trained", False):
            st.info("Train models first in the Train & Tune tab.")
        else:
            le = ss.get("label_encoder", None)
            X_train = ss["processed_data"]["X_train"]
            X_test = ss["processed_data"]["X_test"]
            y_train = ss["processed_data"]["y_train"]
            y_test = ss["processed_data"]["y_test"]
            
            # ---------- Set consistent figure size ----------
            fig_size = (5, 4)

            for model_name, model in ss["trained_models"].items():
                st.write(f"#### {model_name} - Classification Report")
                st.json(ss["results"][model_name])

                # ---------- Collect plots dynamically ----------
                plots = []

                # Confusion Matrix
                try:
                    fig_cm, ax_cm = plt.subplots(figsize=fig_size)
                    ConfusionMatrixDisplay(
                        ss["cmats"][model_name],
                        display_labels=le.classes_ if le else None
                    ).plot(ax=ax_cm, cmap="Blues", colorbar=True)
                    ax_cm.set_title("Confusion Matrix")
                    plots.append(fig_cm)
                except Exception:
                    pass

                # Log/Loss Curve
                try:
                    fig_lc, ax_lc = plt.subplots(figsize=fig_size)
                    plotted = False
                    if hasattr(model, "evals_result"):
                        evals = model.evals_result()
                        key = list(evals.keys())[0]
                        metric_name = list(evals[key].keys())[0]
                        train_metric = evals["validation_0"][metric_name]
                        test_metric = evals["validation_1"][metric_name]
                        ax_lc.plot(train_metric, label=f"Train {metric_name}")
                        ax_lc.plot(test_metric, label=f"Test {metric_name}")
                        plotted = True
                    elif hasattr(model, "staged_predict_proba"):
                        train_losses, test_losses = [], []
                        for proba in model.staged_predict_proba(X_train):
                            train_losses.append(log_loss(y_train, proba, labels=list(range(len(le.classes_)))))
                        for proba in model.staged_predict_proba(X_test):
                            test_losses.append(log_loss(y_test, proba, labels=list(range(len(le.classes_)))))
                        ax_lc.plot(train_losses, label="Train log loss")
                        ax_lc.plot(test_losses, label="Test log loss")
                        plotted = True
                    if plotted:
                        ax_lc.set_title("Log/Loss Curve")
                        ax_lc.set_xlabel("Iterations / Rounds")
                        ax_lc.set_ylabel("Metric")
                        ax_lc.legend()
                        plots.append(fig_lc)
                    else:
                        plt.close(fig_lc)
                except Exception:
                    pass

                # Residuals
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)
                        if len(le.classes_) == 2:
                            true_prob = proba[:, 1]
                            residuals = y_test.astype(float) - true_prob
                        else:
                            idx = np.arange(len(y_test))
                            true_prob = proba[idx, y_test]
                            residuals = 1.0 - true_prob

                        # Residuals Distribution
                        fig_res1, ax_res1 = plt.subplots(figsize=fig_size)
                        sns.histplot(residuals, kde=True, ax=ax_res1)
                        ax_res1.set_title("Residuals Distribution")
                        plots.append(fig_res1)

                        # Residuals vs Predicted
                        fig_res2, ax_res2 = plt.subplots(figsize=fig_size)
                        ax_res2.scatter(true_prob, residuals, alpha=0.4)
                        ax_res2.axhline(0, color="red", linestyle="--", linewidth=1)
                        ax_res2.set_xlabel("Predicted Probability (True Class)")
                        ax_res2.set_ylabel("Residual")
                        ax_res2.set_title("Residuals vs Predicted")
                        plots.append(fig_res2)
                except Exception:
                    pass

                # ROC Curve (binary only)
                try:
                    if hasattr(model, "predict_proba") and len(le.classes_) == 2:
                        proba = model.predict_proba(X_test)[:, 1]
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(y_test, proba)
                        roc_auc = auc(fpr, tpr)
                        fig_roc, ax_roc = plt.subplots(figsize=fig_size)
                        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
                        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('ROC Curve')
                        ax_roc.legend(loc="lower right")
                        plots.append(fig_roc)
                except Exception:
                    pass

                # Feature Importance
                try:
                    if hasattr(model, "feature_importances_"):
                        fig_imp, ax_imp = plt.subplots(figsize=fig_size)
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        sns.barplot(x=importances[indices], y=np.array(ss["processed_data"]["feature_names"])[indices], ax=ax_imp)
                        ax_imp.set_title("Feature Importance")
                        plots.append(fig_imp)
                except Exception:
                    pass

                # ---------- Display plots in 2-column layout ----------
                n = len(plots)
                # If odd, add placeholder to keep layout consistent
                if n % 2 != 0:
                    plots.append(None)

                for i in range(0, len(plots), 2):
                    col1, col2 = st.columns(2)
                    # Left plot
                    if plots[i] is not None:
                        col1.pyplot(plots[i])
                        plt.close(plots[i])
                    else:
                        col1.empty()
                    # Right plot
                    if plots[i+1] is not None:
                        col2.pyplot(plots[i+1])
                        plt.close(plots[i+1])
                    else:
                        col2.empty()


            # ---------------- Ensemble Models ----------------
            if ss.get("ensemble_models", {}):
                st.write("## Ensemble Models Results (Tabular Only)")
                for model_name, model in ss["ensemble_models"].items():
                    st.write(f"#### {model_name} - Ensemble ({selected_ensemble})")
                    
                    # Generate classification report for ensemble models
                    X_test = ss["processed_data"]["X_test"]
                    y_test = ss["processed_data"]["y_test"]
                    le = ss.get("label_encoder", None)
                    
                    y_pred = model.predict(X_test)
                    if le:
                        y_test_labels = le.inverse_transform(y_test)
                        y_pred_labels = le.inverse_transform(y_pred)
                    else:
                        y_test_labels = y_test
                        y_pred_labels = y_pred

                    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
                    st.json(report)

            # ---------- Model Comparison ----------
            st.write("## Model Comparison")
            if ss.get("results", {}):
                metrics = ["precision", "recall", "f1-score"]
                comparison = pd.DataFrame(
                    {m: {metric: ss["results"][m]["weighted avg"][metric] for metric in metrics} for m in ss["results"]}
                ).T
                st.dataframe(comparison)
                st.bar_chart(comparison)
            else:
                st.info("No results to compare yet.")

        # ------------------ Explanation (SHAP) ------------------
        with explain_tab:
            if not ss.get("trained", False):
                st.info("Train models first in the Train & Tune tab.")
            else:
                try:
                    shap = importlib.import_module("shap")
                    st.subheader("SHAP Explainability", help="Visualize how each feature contributes to the model's prediction for a single instance. See which factors pushed the result toward Class 0 or Class 1 for greater interpretability and trust.")

                    le = ss.get("label_encoder", None)
                    classes = list(le.classes_) if le else [0, 1]

                    # Default class index
                    if 1 in classes:
                        default_idx = classes.index(1)
                    elif "1" in classes:
                        default_idx = classes.index("1")
                    else:
                        default_idx = 1 if len(classes) == 2 else 0

                    selected_class = st.selectbox(
                        "Class to explain (applies to all models)",
                        classes,
                        index=default_idx,
                        key="shap-global-class",
                        help="Select which model prediction to explain. Class 0 represents [negative], Class 1 represents [positive]."
                    )
                    class_index = classes.index(selected_class)

                    # Sample data for speed
                    X_train, X_test, y_train, y_test = ss["splits"]
                    feature_names = ss.get("feature_names", X_test.columns.tolist())
                    sample_size = min(200, len(X_test))
                    X_sample = X_test.iloc[:sample_size][feature_names]

                    tree_based = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)
                    fig_size = (5, 4)  # consistent size for all SHAP plots

                    for model_name, model in ss["trained_models"].items():
                        st.write(f"#### {model_name} - SHAP summary plot")
                        cache_key = f"{model_name}_class{class_index}"
                        if cache_key not in ss.get("shap_cache", {}):
                            with st.spinner(f"Computing SHAP values for {model_name}..."):
                                is_tree_model = isinstance(model, tree_based) or model_name in ["XGBoost", "LightGBM"]
                                explainer = shap.TreeExplainer(model) if is_tree_model else shap.Explainer(model, X_train)
                                shap_values = explainer(X_sample)

                                # Normalize to 2D [n_samples, n_features]
                                if isinstance(shap_values, list):
                                    values_to_plot = shap_values[class_index]
                                elif hasattr(shap_values, "values"):
                                    v = shap_values.values
                                    if v.ndim == 3:
                                        values_to_plot = v[:, :, class_index]
                                    else:
                                        values_to_plot = v
                                else:
                                    v = np.array(shap_values)
                                    values_to_plot = v[:, :, class_index] if v.ndim == 3 else v

                                if "shap_cache" not in ss:
                                    ss["shap_cache"] = {}
                                ss["shap_cache"][cache_key] = {"values": values_to_plot, "X_sample": X_sample}

                        vals = ss["shap_cache"][cache_key]["values"]
                        Xs = ss["shap_cache"][cache_key]["X_sample"]

                        # ---------- Plot with consistent size ----------
                        fig_shap, ax_shap = plt.subplots(figsize=fig_size)
                        shap.summary_plot(vals, Xs, show=False, max_display=15)
                        st.pyplot(fig_shap)
                        plt.close(fig_shap)

                except Exception as e:
                    st.info(f"SHAP not available: {e}")


    # ------------------ Live Prediction ------------------
    with predict_tab:
        st.subheader("Live Prediction", help="Get a real-time model prediction by inputting your own data. Enter the feature values below and see the instant outcome and probability score.")
        if not ss["trained_models"]:
            st.info("Train at least one model in the Train & Tune tab to enable live predictions.")
        else:
            feature_names = ss.get("feature_names")
            X_train, X_test, y_train, y_test = ss["splits"]
            X_ref = pd.concat([X_train, X_test], axis=0)
            if feature_names is None:
                feature_names = X_ref.columns.tolist()

            # Dynamic input form
            with st.form("live_pred_form"):
                inputs = {}
                cols = st.columns(3)
                col_index = 0
                for feature in feature_names:
                    unit = feature_units.get(feature, "")
                    label = f"{feature} ({unit})" if unit else feature

                    series = X_ref[feature] if feature in X_ref.columns else pd.Series(dtype=float)
                    current_col = cols[col_index % 3]
                    col_index += 1

                    if is_bool_dtype(series):
                        default_val = bool(series.mode().iloc[0]) if not series.mode().empty else False
                        inputs[feature] = current_col.checkbox(label, value=default_val, key=f"live_{feature}_bool")
                    elif is_numeric_dtype(series):
                        series_non_na = series.dropna()
                        if series_non_na.empty:
                            min_val = 0.0; max_val = 1.0; default_val = 0.0
                        else:
                            min_val = float(np.nanmin(series_non_na))
                            max_val = float(np.nanmax(series_non_na))
                            default_val = float(np.nanmedian(series_non_na))

                        is_int = pd.api.types.is_integer_dtype(series)
                        if is_int:
                            min_cast = int(np.floor(min_val)) if np.isfinite(min_val) else -10**9
                            max_cast = int(np.ceil(max_val)) if np.isfinite(max_val) else 10**9
                            default_cast = int(np.round(default_val)) if np.isfinite(default_val) else 0
                            step_val = 1
                            inputs[feature] = current_col.number_input(
                                label, value=default_cast,
                                min_value=min_cast, max_value=max_cast,
                                step=step_val, key=f"live_{feature}_int"
                            )
                        else:
                            if not (np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val):
                                min_val, max_val = -1e9, 1e9
                            step_val = (max_val - min_val) / 100.0
                            inputs[feature] = current_col.number_input(
                                label,
                                value=float(default_val) if np.isfinite(default_val) else 0.0,
                                min_value=float(min_val), max_value=float(max_val),
                                step=float(step_val), format="%f", key=f"live_{feature}_float"
                            )
                    else:
                        options = [str(x) for x in series.dropna().unique().tolist()]
                        default_option = options[0] if options else ""
                        inputs[feature] = current_col.selectbox(
                            label, options=options or [""], index=0, key=f"live_{feature}_cat"
                        )

                submitted = st.form_submit_button("Predict")

            if submitted:
                input_row = {f: inputs.get(f, None) for f in feature_names}
                input_df = pd.DataFrame([input_row])

                # Coerce numeric features back to numeric
                for f in feature_names:
                    if f in X_ref.columns and is_numeric_dtype(X_ref[f]):
                        input_df[f] = pd.to_numeric(input_df[f], errors="coerce")

                trained = {k: v for k, v in ss["trained_models"].items() if k in selected_models} or ss["trained_models"]
                le = ss["label_encoder"]
                classes = list(le.classes_)

                summary_rows = []
                expanders = []
                for mname, mdl in trained.items():
                    try:
                        if hasattr(mdl, "predict_proba"):
                            proba = mdl.predict_proba(input_df)
                            pred_idx = int(np.argmax(proba[0]))
                            pred_lbl = le.inverse_transform([pred_idx])[0]
                            conf = float(proba[0][pred_idx])
                            prob_df = pd.DataFrame({"Class": classes, "Probability": proba[0]})
                        else:
                            pred_idx = int(mdl.predict(input_df)[0])
                            pred_lbl = le.inverse_transform([pred_idx])[0]
                            conf = np.nan
                            prob_df = pd.DataFrame({"Class": classes, "Probability": [np.nan] * len(classes)})

                        summary_rows.append({"Model": mname, "Prediction": pred_lbl, "Confidence": conf})
                        expanders.append((mname, prob_df))
                    except Exception as e:
                        summary_rows.append({"Model": mname, "Prediction": f"Error: {e}", "Confidence": np.nan})

                st.write("#### Predictions")
                st.dataframe(pd.DataFrame(summary_rows))

                for mname, prob_df in expanders:
                    with st.expander(f"{mname} - Class probabilities"):
                        st.dataframe(prob_df)

else:
    st.info("Please upload a dataset to continue.")
