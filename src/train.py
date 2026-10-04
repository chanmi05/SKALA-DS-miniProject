"""모델 학습 · 선택 · 평가 (Day 2).

설계는 Day 1 보고서의 모델 전략을 그대로 구현한다.
- 타깃 : log10(cycle_life) → 예측 후 10**ŷ 로 되돌려 MAPE 계산
- 분할 : Batch 1 = Train(정책 단위 GroupKFold CV) + Valid(정책 그룹 단위 hold-out, 약 25%)
         Batch 2 = Test  (Batch 3는 EDA 비교에만 사용, 테스트하지 않음)
- 선택 규칙 : ① CV MAPE 최소 ② Train–Valid 차이가 큰 모델(과적합) 제외 ③ 1%p 이내면 단순한 모델
             ④ Batch 2 성능은 선택에 쓰지 않는다 (보고용)

사용 예
    python -m src.train            # 전체 실행 → results/ 에 결과 저장
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor

from .utils import PROCESSED_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')

SEED = 42
TARGET_MAPE = 9.1            # 원논문 Regression 성능 (%)
N_FOLDS = 5
HOLDOUT_FRAC = 0.25
OVERFIT_GAP = 5.0            # Train(CV) 대비 Valid MAPE가 이만큼(%p) 이상 나쁘면 과적합으로 보고 제외
TIE_MARGIN = 1.0             # CV MAPE 차이가 이 이내면 더 단순한 모델 선택

# 단순한 것부터 (선택 규칙 ③의 순서)
COMPLEXITY = ['Linear (1 feat)', 'Ridge', 'Lasso', 'ElasticNet', 'PLS', 'GPR', 'SVR', 'RandomForest', 'XGBoost']


# ----------------------------------------------------------------------------- metrics
def mape(y_true_life, y_pred_life):
    y_true_life, y_pred_life = np.ravel(y_true_life), np.ravel(y_pred_life)
    return float(np.mean(np.abs(y_pred_life - y_true_life) / y_true_life) * 100)


def mape_from_log(y_true_log, y_pred_log):
    return mape(10 ** np.ravel(y_true_log), 10 ** np.ravel(y_pred_log))


def rmse(y_true_life, y_pred_life):
    return float(np.sqrt(np.mean((np.ravel(y_pred_life) - np.ravel(y_true_life)) ** 2)))


MAPE_SCORER = make_scorer(mape_from_log, greater_is_better=False)


# ----------------------------------------------------------------------------- data
def load_features():
    """02_feature_engineering 출력(features.csv, feature_sets.json)을 읽는다. 없으면 직접 만든다."""
    fpath = os.path.join(PROCESSED_DIR, 'features.csv')
    spath = os.path.join(PROCESSED_DIR, 'feature_sets.json')
    if os.path.exists(fpath) and os.path.exists(spath):
        F = pd.read_csv(fpath, index_col=0)
        sets = json.load(open(spath))['sets']
    else:
        from .preprocess import get_dataset
        from .features import build_feature_table, FEATURE_SETS
        F = build_feature_table(get_dataset())
        sets = dict(FEATURE_SETS)
    if 'policy_base' not in F:
        F['policy_base'] = F['policy'].str.replace('-newstructure', '', case=False)
    F['structure'] = np.where(F['policy'].str.contains('newstructure', case=False), 'new', 'old')
    F['log_life'] = np.log10(F['cycle_life'])
    return F, sets


def split_batch1(F, frac=HOLDOUT_FRAC, seed=SEED):
    """Batch 1을 정책 그룹 단위로 Train / Valid 분할.

    같은 정책 셀은 반드시 같은 쪽에 두고(누수 방지), 정책 그룹을 평균 수명 순으로 정렬해
    구간마다 하나씩 뽑아 Valid에 짧은·긴 수명이 고르게 들어가도록 층화한다.
    """
    b1 = F[F.batch == 'B1']
    g = b1.groupby('policy_base').cycle_life.mean().sort_values()
    n_valid = max(1, int(round(len(g) * frac)))
    rng = np.random.default_rng(seed)
    bins = np.array_split(g.index.to_numpy(), n_valid)
    valid_groups = [rng.choice(bin_) for bin_ in bins]
    is_valid = b1.policy_base.isin(valid_groups)
    return b1[~is_valid].copy(), b1[is_valid].copy()


# ----------------------------------------------------------------------------- models
def _pipe(model):
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', model)])


class PLS1D(PLSRegression):
    """PLSRegression의 2차원 출력을 1차원으로 (GridSearch·평가 함수와 맞추기 위해)."""
    def predict(self, X, copy=True):
        return np.ravel(super().predict(X, copy=copy))


def model_zoo():
    """(이름, 파이프라인, 하이퍼파라미터 격자). 격자는 Day 1 보고서의 설정과 같다."""
    alphas_en = np.logspace(-4, 1, 30)
    zoo = [
        ('Linear (1 feat)', _pipe(LinearRegression()), {}),
        ('Ridge', _pipe(Ridge(random_state=SEED)), {'model__alpha': np.logspace(-3, 3, 25)}),
        ('Lasso', _pipe(Lasso(max_iter=100000, random_state=SEED)), {'model__alpha': alphas_en}),
        ('ElasticNet', _pipe(ElasticNet(max_iter=100000, random_state=SEED)),
         {'model__alpha': alphas_en, 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]}),
        ('PLS', _pipe(PLS1D(scale=False)), {'model__n_components': [1, 2, 3, 4, 5]}),
        ('SVR', _pipe(SVR()), {'model__C': [0.1, 1, 10, 100], 'model__epsilon': [0.01, 0.05, 0.1],
                               'model__gamma': ['scale', 0.01, 0.1, 1]}),
        ('GPR', _pipe(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
            normalize_y=True, n_restarts_optimizer=3, random_state=SEED)), {}),
        ('RandomForest', _pipe(RandomForestRegressor(random_state=SEED, n_jobs=-1)),
         {'model__max_depth': [2, 3, 4], 'model__n_estimators': [200, 500], 'model__min_samples_leaf': [3, 5]}),
    ]
    try:
        from xgboost import XGBRegressor
        zoo.append(('XGBoost', _pipe(XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0)),
                    {'model__max_depth': [2, 3, 4], 'model__n_estimators': [200, 500],
                     'model__learning_rate': [0.03, 0.1]}))
    except ImportError:
        print('xgboost가 없어 XGBoost는 건너뜁니다 (pip install xgboost)')
    return zoo


# ----------------------------------------------------------------------------- fit / evaluate
def fit_one(name, pipe, grid, feats, train, n_folds=N_FOLDS):
    """정책 단위 GroupKFold로 하이퍼파라미터 탐색. 반환 : 최적 파이프라인, CV MAPE, CV 예측(out-of-fold)."""
    X, y, groups = train[feats], train['log_life'], train['policy_base']
    cv = GroupKFold(n_splits=min(n_folds, groups.nunique()))
    gs = GridSearchCV(pipe, grid or {}, scoring=MAPE_SCORER, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y, groups=groups)
    # 최적 설정의 out-of-fold 예측 (그림·Train 성능용)
    oof = np.zeros(len(train))
    for tr, va in cv.split(X, y, groups):
        est = _clone(gs.best_estimator_).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = np.ravel(est.predict(X.iloc[va]))
    return gs.best_estimator_, -gs.best_score_, oof, gs.best_params_


def _clone(est):
    from sklearn.base import clone
    return clone(est)


def evaluate(est, df, feats):
    pred_log = np.ravel(est.predict(df[feats]))
    pred = 10 ** pred_log
    return pred, mape(df['cycle_life'], pred), rmse(df['cycle_life'], pred)


def compare_models(F, sets, train, valid, main_set='selected', extra_sets=('variance', 'discharge', 'full'),
                   test_batches=('B2',), verbose=True):
    """모든 후보 모델 × 피처 세트를 학습·평가하고 결과 표를 반환한다."""
    rows, fitted = [], {}
    runs = []
    for name, pipe, grid in model_zoo():
        fs = 'variance' if name == 'Linear (1 feat)' else main_set
        runs.append((name, pipe, grid, fs))
    # Elastic Net은 피처 세트 비교도 함께 (Q1·C2 제외 ablation 포함)
    en = [z for z in model_zoo() if z[0] == 'ElasticNet'][0]
    for fs in extra_sets:
        runs.append(('ElasticNet', en[1], en[2], fs))
    if 'selected' in sets:
        sets = dict(sets)
        sets['selected-noQ1C2'] = [f for f in sets['selected'] if f not in ('Q1', 'C2')]
        runs.append(('ElasticNet', en[1], en[2], 'selected-noQ1C2'))

    for name, pipe, grid, fs in runs:
        feats = sets[fs]
        best, cv_mape, oof, params = fit_one(name, _clone(pipe), grid, feats, train)
        _, va_mape, va_rmse = evaluate(best, valid, feats)
        row = dict(model=name, feature_set=fs, n_feats=len(feats), cv_mape=cv_mape, valid_mape=va_mape,
                   gap_train_valid=va_mape - cv_mape, params=json.dumps({k.replace('model__', ''): (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in params.items()}))
        for b in test_batches:
            d = F[F.batch == b]
            if len(d):
                _, m, r = evaluate(best, d, feats)
                row[f'{b}_mape'] = m          # 보고용 (선택에는 쓰지 않음)
        rows.append(row)
        fitted[(name, fs)] = dict(est=best, feats=feats, oof=oof, params=params)
        if verbose:
            print(f'{name:16s} [{fs:16s}] CV {cv_mape:5.1f}%  Valid {va_mape:5.1f}%  '
                  + '  '.join(f'{b} {row.get(f"{b}_mape", np.nan):5.1f}%' for b in test_batches))
    res = pd.DataFrame(rows)
    return res, fitted, sets


def select_model(res):
    """선택 규칙 ①~③ (Batch 2 성능은 보지 않는다)."""
    cand = res[res.gap_train_valid <= OVERFIT_GAP].copy()
    if cand.empty:
        cand = res.copy()
    best_cv = cand.cv_mape.min()
    near = cand[cand.cv_mape <= best_cv + TIE_MARGIN].copy()
    near['complexity'] = near.model.map({m: i for i, m in enumerate(COMPLEXITY)})
    near = near.sort_values(['complexity', 'n_feats', 'cv_mape'])
    return near.iloc[0], cand


def refit_full_b1(fitted_entry, b1):
    """선택한 설정(하이퍼파라미터 고정)으로 Batch 1 전체(41셀)를 다시 학습 → Test 평가용."""
    est = _clone(fitted_entry['est'])
    est.set_params(**fitted_entry['params'])
    return est.fit(b1[fitted_entry['feats']], b1['log_life'])


def performance_table(cv_mape, valid_mape, test_mape, b3_mape=None, target=TARGET_MAPE):
    rows = [('Train (Batch 1 CV)', cv_mape, ''),
            ('Valid (Batch 1 Hold-out)', valid_mape, ''),
            ('Test (Batch 2)', test_mape, ''),
            ('Gap (Train-Valid)', valid_mape - cv_mape, '(+) : 과적합 의심'),
            ('Gap (Valid-Test)', test_mape - valid_mape, '(+) : 배치간 일반화 저하 의심'),
            ('Gap (Target-Test)', test_mape - target, f'Target : 원논문 {target}%')]
    if b3_mape is not None:
        rows += [('Test (Batch 3)', b3_mape, 'additional'),
                 ('Gap (Batch2-Batch3)', test_mape - b3_mape, 'Test 성능 간 비교'),
                 ('Gap (Target-Test, Batch 3)', b3_mape - target, 'Batch 3 기준, 원논문 성능 비교')]
    return pd.DataFrame(rows, columns=['구분', 'MAPE (%)', '비고'])


def coefficients(est, feats):
    m = est.named_steps['model']
    if hasattr(m, 'coef_'):
        return pd.Series(np.ravel(m.coef_), index=feats).sort_values(key=np.abs, ascending=False)
    return None


# ----------------------------------------------------------------------------- main
def run(verbose=True):
    F, sets = load_features()
    b1 = F[F.batch == 'B1']
    train, valid = split_batch1(F)
    res, fitted, sets = compare_models(F, sets, train, valid, verbose=verbose)
    chosen, _ = select_model(res)
    entry = fitted[(chosen.model, chosen.feature_set)]
    final = refit_full_b1(entry, b1)
    out = {}
    for b in ['B2']:
        d = F[F.batch == b]
        if len(d):
            out[b] = evaluate(final, d, entry['feats'])
    perf = performance_table(chosen.cv_mape, chosen.valid_mape, out['B2'][1])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    perf.to_csv(os.path.join(RESULTS_DIR, 'model_performance.csv'), index=False, encoding='utf-8-sig')
    res.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    return dict(F=F, sets=sets, train=train, valid=valid, res=res, fitted=fitted, chosen=chosen,
                entry=entry, final=final, test=out, perf=perf)


if __name__ == '__main__':
    r = run()
    print('\n선택 :', r['chosen'].model, '/', r['chosen'].feature_set)
    print(r['perf'].to_string(index=False))
