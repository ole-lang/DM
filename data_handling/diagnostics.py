import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Optional plotting libraries: import if available, otherwise define fallbacks
HAS_MATPLOTLIB = True
HAS_SEABORN = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
except Exception:
    HAS_SEABORN = False


def _unwrap_model(model):
    """Versuche, ein sklearn-ähnliches Estimator-Objekt aus Wrappern zu extrahieren.
    Falls nicht möglich, wird das Originalobjekt zurückgegeben.
    """
    # häufige Wrapper-Attribute; wir unwrappen NICHT das Attribut 'model',
    # weil viele Wrapper (z.B. Keras-Wrapper) ein internes .model besitzen
    # das nicht als sklearn-estimator für learning_curve/cv geeignet ist.
    for attr in ("estimator", "clf", "sklearn_model", "est"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            return inner
    return model


class SklearnAdapter:
    """Adapter, der ein lokales Modell mit train()/predict() in ein sklearn-like Objekt umwandelt.
    Nötig, weil die Projekt-Modelle eigene .train() APIs haben.
    """
    def __init__(self, model=None):
        # Default None allows sklearn.clone to create a fresh instance; get_params will ensure
        # cloning passes the original model object through if available.
        self._model = model

    def fit(self, X, y, **kwargs):
        # Versuche train, sonst fit
        if hasattr(self._model, 'train'):
            self._model.train(X, y)
        elif hasattr(self._model, 'fit'):
            self._model.fit(X, y)
        else:
            raise AttributeError("Wrapped model hat weder train noch fit")
        return self

    def predict(self, X):
        if hasattr(self._model, 'predict'):
            return self._model.predict(X)
        raise AttributeError("Wrapped model hat keine predict-Methode")

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))

    def get_params(self, deep=True):
        # Damit sklearn.clone() das wrapped model korrekt rekonstruiert, geben
        # wir das zugrundeliegende Modell als Parameter zurück.
        return {"model": self._model}

    def set_params(self, **params):
        # Akzeptiere set_params von sklearn; falls 'model' übergeben wird, speichern
        if 'model' in params:
            self._model = params['model']
        return self


def inspect_target(y):
    """Drucke Basisstatistiken und zeige Histogram + Boxplot der Zielvariable."""
    if isinstance(y, (pd.Series, np.ndarray)):
        ys = pd.Series(y)
    else:
        ys = pd.Series(list(y))
    print("Target: count=", len(ys))
    print("mean=", ys.mean(), "std=", ys.std(), "skew=", ys.skew(), "kurtosis=", ys.kurtosis())
    if HAS_SEABORN and HAS_MATPLOTLIB:
        sns.histplot(ys, kde=True, bins=60)
        plt.title("Target distribution")
        plt.show()
        sns.boxplot(x=ys)
        plt.title("Target boxplot")
        plt.show()
    elif HAS_MATPLOTLIB:
        plt.hist(ys, bins=60)
        plt.title("Target distribution (matplotlib)")
        plt.show()
    else:
        print("Keine plotting-Bibliothek installiert (matplotlib/seaborn). Nur Statistiken angezeigt.")


def safe_predict(model, X):
    """Sichere Vorhersage: versucht verschiedene predict-Varianten und gibt None bei Fehler zurück."""
    try:
        return model.predict(X)
    except Exception:
        # eventuell Wrapper hat predict_proba oder predict_generator (bei Keras) - ignorieren
        try:
            # einige Keras-Wrapper nutzen predict on model.model
            inner = _unwrap_model(model)
            return inner.predict(X)
        except Exception as e:
            print("safe_predict: konnte nicht vorhersagen:", e)
            return None


def print_train_test_scores(model, X_train, X_test, y_train, y_test, name="model"):
    """Gibt R2 und RMSE für Train und Test aus; versucht nicht, neu zu trainieren."""
    est = _unwrap_model(model)
    # Versuch Vorhersagen auf beiden Sets
    y_pred_train = safe_predict(model, X_train)
    y_pred_test = safe_predict(model, X_test)
    if y_pred_train is None or y_pred_test is None:
        print(f"{name}: Vorhersage fehlgeschlagen (evtl. inkompatibler Wrapper). Versuche safe_r2 andernorts.")
        return
    try:
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    except Exception:
        # Falls die sklearn-Version oder ein Namenskonflikt Probleme macht, berechne manuell
        try:
            mse = float(((np.array(y_test) - np.array(y_pred_test)) ** 2).mean())
            rmse_test = float(np.sqrt(mse))
        except Exception as e:
            print(f"Fehler bei RMSE-Berechnung für {name}: {e}")
            rmse_test = None
    try:
        r2_tr = float(r2_score(y_train, y_pred_train))
    except Exception as e:
        print(f"Fehler bei r2_score train für {name}: {e}")
        r2_tr = None
    try:
        r2_te = float(r2_score(y_test, y_pred_test))
    except Exception as e:
        print(f"Fehler bei r2_score test für {name}: {e}")
        r2_te = None
    print(f"{name} -> R2 train: {r2_tr}, test: {r2_te}; RMSE test: {rmse_test}")


def print_feature_importances(model, feature_names=None, top=20):
    """Druckt Feature Importances falls verfügbar; versucht, gängige Attribute zu finden."""
    est = _unwrap_model(model)
    try:
        # Direkt vorhandene Importances
        if hasattr(est, "feature_importances_"):
            imps = pd.Series(est.feature_importances_, index=feature_names)
        elif hasattr(est, "coef_"):
            # lineare Modelle
            coefs = np.ravel(est.coef_)
            imps = pd.Series(np.abs(coefs), index=feature_names)
        else:
            # Versuch, intern .model zu unwrappen (z.B. RandomForestModel.model)
            try:
                inner = getattr(model, 'model', None)
                if inner is not None and hasattr(inner, 'feature_importances_'):
                    imps = pd.Series(inner.feature_importances_, index=feature_names)
                elif inner is not None and hasattr(inner, 'coef_'):
                    imps = pd.Series(np.abs(np.ravel(inner.coef_)), index=feature_names)
                else:
                    print("Keine feature_importances_ oder coef_ im Modell gefunden.")
                    return
            except Exception:
                print("Keine feature_importances_ oder coef_ im Modell gefunden.")
                return
        imps = imps.sort_values(ascending=False)
        print(imps.head(top))
        if HAS_SEABORN and HAS_MATPLOTLIB:
            sns.barplot(x=imps.head(top).values, y=imps.head(top).index)
            plt.title("Top feature importances")
            plt.show()
        elif HAS_MATPLOTLIB:
            plt.barh(imps.head(top).index, imps.head(top).values)
            plt.title("Top feature importances")
            plt.show()
    except Exception as e:
        print("Fehler beim Ermitteln der Feature Importances:", e)


def plot_residuals(y_true, y_pred, title="Residuals"):
    """Einfacher Residualplot"""
    if y_pred is None:
        print("plot_residuals: y_pred ist None")
        return
    res = np.array(y_true) - np.array(y_pred)
    if not HAS_MATPLOTLIB:
        print("plot_residuals: matplotlib nicht verfügbar, überspringe Plot.")
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, res, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("y_pred"); plt.ylabel("residual"); plt.title(title)
    plt.show()


def plot_learning_curve_estimator(estimator, X, y, scoring='r2', cv=5):
    """Versucht, learning_curve zu benutzen. Falls estimator inkompatibel ist, bricht Function ab."""
    est = _unwrap_model(estimator)
    # Falls kein sklearn-like fit vorhanden, wrappe
    est_for_sklearn = est if hasattr(est, 'fit') else SklearnAdapter(estimator)
    try:
        train_sizes, train_scores, val_scores = learning_curve(est_for_sklearn, X, y, cv=cv, scoring=scoring,
                                                               train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1)
    except Exception as e:
        print("plot_learning_curve_estimator: konnte learning_curve nicht verwenden:", e)
        return
    if not HAS_MATPLOTLIB:
        print("plot_learning_curve_estimator: matplotlib nicht installiert, zeige Zahlen.")
        print('train_sizes:', train_sizes)
        print('train_scores_mean:', train_scores.mean(axis=1))
        print('val_scores_mean:', val_scores.mean(axis=1))
        return
    plt.plot(train_sizes, train_scores.mean(axis=1), label='train')
    plt.plot(train_sizes, val_scores.mean(axis=1), label='cv')
    plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1), val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.legend(); plt.xlabel('train size'); plt.ylabel(scoring); plt.title('Learning curve'); plt.show()


def cv_scores(estimator, X, y, cv=5, scoring='r2'):
    est = _unwrap_model(estimator)
    # cross_val_score benötigt sklearn-like estimator mit fit; wrappe falls nötig
    est_for_sklearn = est if hasattr(est, 'fit') else SklearnAdapter(estimator)
    try:
        scores = cross_val_score(est_for_sklearn, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"CV {cv}-fold {scoring}: mean={scores.mean():.4f}, std={scores.std():.4f}, all={scores}")
        return scores
    except Exception as e:
        print("cv_scores: cross_val_score fehlgeschlagen:", e)
        return None


def test_mlp_scaling(mlp_model, X_train, X_test, y_train, y_test):
    """Erstellt Pipeline StandardScaler + MLP (falls kompatibel) und trainiert kurz.
    Funktion ist vorsichtig: nur wenn das unwrapped Modell ein sklearn-like Estimator ist.
    """
    est = _unwrap_model(mlp_model)
    try:
        # Prüfen, ob estimator fit/predict besitzt
        if not (hasattr(est, 'fit') and hasattr(est, 'predict')):
            print("test_mlp_scaling: der gegebene MLP ist kein sklearn-Estimator; überspringe.")
            return
        pipe = make_pipeline(StandardScaler(), est)
        print("Training MLP mit StandardScaler in Pipeline (kann lange dauern)...")
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)
        print(f"MLP(scaled) -> R2 test: {r2_score(y_test, y_pred_test):.4f}")
    except Exception as e:
        print("test_mlp_scaling fehlgeschlagen:", e)


def print_prediction_stats(model, X, y, name="model", top_n=10):
    """Druckt einfache Statistiken zu Vorhersagen und größten Residuen (ohne Plots)."""
    try:
        y_pred = safe_predict(model, X)
        if y_pred is None:
            print(f"{name}: Vorhersage fehlgeschlagen, keine Stats.")
            return
        y = np.array(y)
        y_pred = np.array(y_pred)
        res = y - y_pred
        print(f"{name} predictions: count={len(y_pred)}, mean={y_pred.mean():.3f}, median={np.median(y_pred):.3f}, min={y_pred.min():.3f}, max={y_pred.max():.3f}")
        for p in (1,5,25,50,75,95,99):
            print(f" pred p{p}: {np.percentile(y_pred, p):.3f}")
        print(f"residuals: mean={res.mean():.3f}, std={res.std():.3f}")
        worst_idx = np.argsort(np.abs(res))[::-1][:top_n]
        print(f"Top {top_n} errors (actual, pred, abs_err):")
        for i in worst_idx:
            print(y[i], y_pred[i], abs(res[i]))
    except Exception as e:
        print(f"print_prediction_stats fehlgeschlagen für {name}: {e}")


def inspect_extreme_predictions(model, X, y, top_n=10, residual_multiplier=3.0):
    """Zeigt die Datenpunkte mit den größten absoluten Residuen und prüft NaN/Inf in ihren Features.

    - residual_multiplier: markiert alle Residuen größer als residual_multiplier * std(residuals)
    """
    try:
        y_pred = safe_predict(model, X)
        if y_pred is None:
            print("inspect_extreme_predictions: Vorhersage fehlgeschlagen")
            return
        y = np.array(y)
        y_pred = np.array(y_pred)
        res = y - y_pred
        abs_res = np.abs(res)
        thresh = residual_multiplier * np.std(res)
        idxs_threshold = np.where(abs_res > thresh)[0]
        print(f"inspect_extreme_predictions: residual std={np.std(res):.3f}, threshold={thresh:.3f}, count>{residual_multiplier}*std: {len(idxs_threshold)}")
        # Top N by absolute residual
        worst_idx = np.argsort(abs_res)[::-1][:top_n]
        print(f"Top {top_n} worst indices:", worst_idx)
        # If X is a DataFrame, print rows; else print arrays and NaN/Inf checks
        for i in worst_idx:
            print(f"--- idx={i} actual={y[i]} pred={y_pred[i]:.6f} abs_err={abs_res[i]:.6f}")
            if hasattr(X, 'iloc'):
                row = X.iloc[i]
                print(row.to_dict())
                # NaN/Inf checks
                n_nans = int(row.isna().sum())
                n_infs = int(np.isinf(row.values).sum())
                if n_nans or n_infs:
                    print(f"Row has NaNs: {n_nans}, Infs: {n_infs}")
            else:
                row = np.array(X)[i]
                print(row)
                print("NaNs in row:", np.isnan(row).sum(), "Infs in row:", np.isinf(row).sum())
        # Also print a small sample of threshold-exceeding indices if any
        if len(idxs_threshold) > 0:
            sample = idxs_threshold[:min(10, len(idxs_threshold))]
            print("Sample indices exceeding threshold:", sample)
    except Exception as e:
        print("inspect_extreme_predictions failed:", e)


def run_quick_checks(models: dict, X_train, X_test, y_train, y_test, df_features=None):
    """Orchestrator: führt einige schnelle checks aus. models erwartet ein dict name->model."""
    print("--- inspect target (train) ---")
    inspect_target(y_train)

    print("--- Train/Test scores ---")
    for name, model in models.items():
        try:
            print_train_test_scores(model, X_train, X_test, y_train, y_test, name=name)
        except Exception as e:
            print(f"Fehler bei train/test für {name}:", e)

    # Feature Importances für Baum-Modelle
    if df_features is not None:
        feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        for name, model in models.items():
            print(f"--- feature importances for {name} ---")
            print_feature_importances(model, feature_names)

    # Optional: learning curve for a representative model if available
    for name, model in models.items():
        print(f"--- learning curve attempt for {name} ---")
        plot_learning_curve_estimator(model, X_train, y_train)
        break

    print("--- done quick checks ---")
