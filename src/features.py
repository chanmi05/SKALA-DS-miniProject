"""피처 생성 · 다중공선성 진단 · 열화 곡선 지표.

피처 그룹은 원논문(Severson et al. 2019) Supplementary Table 1의 구성(Discharge 13개 → Full 20개)을 따르고,
과제용으로 충전 정책 수치 피처(policy)를 추가했다. 모든 피처는 **초기 100 사이클 이내 데이터만** 사용한다.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .preprocess import rolling_med

# ----------------------------------------------------------------------------- feature groups
DQ_FEATS = ['dq_log_absmin', 'dq_mean', 'dq_log_var', 'dq_skew', 'dq_kurt', 'dq_2V']
CURVE_FEATS = ['qd_slope_2_100', 'qd_int_2_100', 'qd_slope_91_100', 'qd_int_91_100',
               'qd_2', 'qd_max_minus_2', 'qd_100']
OTHER_FEATS = ['chargetime_5', 'tmax_2_100', 'tmin_2_100', 'tavg_2_100',
               'ir_2', 'ir_min_2_100', 'ir_100_minus_2']
POLICY_FEATS = ['C1', 'Q1', 'C2', 'avgC']

FEATURE_SETS = {
    'variance': ['dq_log_var'],                                   # 논문 'Variance' model
    'discharge': DQ_FEATS + CURVE_FEATS,                          # 논문 'Discharge' model 후보 (13)
    'full': DQ_FEATS + CURVE_FEATS + OTHER_FEATS,                 # 논문 'Full' model 후보 (20)
    'full+policy': DQ_FEATS + CURVE_FEATS + OTHER_FEATS + POLICY_FEATS,
}
ALL_FEATS = FEATURE_SETS['full+policy']


# ----------------------------------------------------------------------------- ΔQ(V)
def dq_curve(qdlin, cid, a=100, b=10):
    """ΔQ_{a-b}(V) = Q_a(V) - Q_b(V) (1000개 전압 격자). 계산 불가 시 None."""
    q = qdlin.get(cid) or {}
    if q.get(a) is None or q.get(b) is None:
        return None
    return q[a] - q[b]


def dq_stats(v):
    return dict(dq_log_absmin=np.log10(np.abs(v.min()) + 1e-12), dq_mean=v.mean(),
                dq_log_var=np.log10(v.var() + 1e-12), dq_skew=stats.skew(v),
                dq_kurt=stats.kurtosis(v), dq_2V=v[-1], dq_var=v.var(), dq_min=v.min())


# ----------------------------------------------------------------------------- capacity curve / others
def _lin(g, lo, hi):
    s = g[(g.cycle >= lo) & (g.cycle <= hi)]
    if len(s) < 3:
        return np.nan, np.nan
    p = np.polyfit(s.cycle, s.QD, 1)
    return p[0], p[1]


def curve_and_other_feats(g):
    """한 셀의 summary(정제 후)에서 사이클 2~100 기반 피처 계산."""
    g = g[(g.cycle >= 2) & (g.cycle <= 100)].sort_values('cycle')
    s1, i1 = _lin(g, 2, 100)
    s2, i2 = _lin(g, 91, 100)
    head, tail = g.head(3), g.tail(3)
    qd2 = head.QD.iloc[0] if len(head) else np.nan
    return pd.Series(dict(
        qd_slope_2_100=s1 * 1e4, qd_int_2_100=i1, qd_slope_91_100=s2 * 1e4, qd_int_91_100=i2,
        qd_2=qd2, qd_max_minus_2=(g.QD.max() - qd2) * 1e3, qd_100=tail.QD.median(),
        chargetime_5=g[g.cycle <= 6].chargetime.median(),
        tmax_2_100=g.Tmax.max(), tmin_2_100=g.Tmin.min(), tavg_2_100=g.Tavg.mean(),
        ir_2=head.IR.median(), ir_min_2_100=g.IR[g.IR > 0].min(),
        ir_100_minus_2=(tail.IR.median() - head.IR.median()) * 1e3))


def build_feature_table(ds, a=100, b=10):
    """셀 × 피처 테이블. index=cell, 열 = batch, cycle_life, log_life, ALL_FEATS, policy."""
    meta, summ = ds['meta'], ds['summ']
    dq = {}
    for c in meta.index:
        v = dq_curve(ds['qdlin'], c, a, b)
        if v is not None:
            dq[c] = dq_stats(v)
    dqf = pd.DataFrame(dq).T
    cf = summ.groupby('cell').apply(curve_and_other_feats)
    F = meta[['batch', 'policy', 'cycle_life'] + POLICY_FEATS].join(dqf).join(cf)
    F['log_life'] = np.log10(F['cycle_life'])
    return F


# ----------------------------------------------------------------------------- correlation / VIF
def corr_by_batch(F, feats, target='log_life', batches=('B1', 'B2', 'B3'), method='pearson'):
    out = {}
    for b in list(batches) + ['ALL']:
        s = F if b == 'ALL' else F[F.batch == b]
        out[b] = s[feats].corrwith(s[target], method=method)
    R = pd.DataFrame(out)
    R['sign_stable'] = np.sign(R[list(batches)]).nunique(axis=1) == 1
    R['min_abs_r'] = R[list(batches)].abs().min(axis=1)
    return R


def vif(df):
    X = (df - df.mean()) / df.std()
    out = {}
    for c in X.columns:
        y, Xo = X[c].values, np.c_[np.ones(len(X)), X.drop(columns=c).values]
        beta = np.linalg.lstsq(Xo, y, rcond=None)[0]
        r2 = 1 - ((y - Xo @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / max(1 - r2, 1e-6)
    return pd.Series(out)


def usable_features(F, feats, batch='B1'):
    s = F[F.batch == batch]
    return [c for c in feats if s[c].notna().mean() > .9 and s[c].std() > 0]


def select_by_vif(train, must_keep=('dq_log_var',), threshold=10):
    """VIF가 가장 큰 피처부터 하나씩 제거 (must_keep은 유지)."""
    keep = list(train.columns)
    while len(keep) > 2:
        v = vif(train[keep])
        cand = v.drop(index=[k for k in must_keep if k in v.index]).sort_values(ascending=False)
        if len(cand) == 0 or cand.iloc[0] < threshold:
            break
        keep.remove(cand.index[0])
    return keep


# ----------------------------------------------------------------------------- degradation descriptors (EDA)
def knee_point(cyc, q):
    """2-구간 선형회귀 SSE가 최소가 되는 분할점 (누적합으로 전체 분할점을 한 번에 계산)."""
    x, y = np.asarray(cyc, float), np.asarray(q, float)

    def sse_prefix(x, y):
        n = np.arange(1, len(x) + 1)
        sx, sy, sxx, sxy, syy = [np.cumsum(a) for a in (x, y, x * x, x * y, y * y)]
        vxx = sxx - sx ** 2 / n
        vxy = sxy - sx * sy / n
        vyy = syy - sy ** 2 / n
        return vyy - np.divide(vxy ** 2, vxx, out=np.zeros_like(vxx), where=vxx > 1e-12)

    left, right = sse_prefix(x, y), sse_prefix(x[::-1], y[::-1])[::-1]
    n = len(x)
    ks = np.arange(max(int(n * .1), 1), int(n * .95))
    return x[ks[np.argmin(left[ks - 1] + right[ks])]]


def degradation_table(meta, summ):
    """셀별 knee point, 수명 구간별 열화 속도(×1e-4 Ah/cycle), 정규화 SOH 곡선.
    EOL까지 곡선이 관측된 셀(curve_complete)만 계산한다."""
    grid = np.linspace(0, 1, 101)
    rows, curves = [], {}
    ok = meta.index[meta.get('curve_complete', pd.Series(True, index=meta.index)).astype(bool)]
    for cid, g in summ[summ.cell.isin(ok)].groupby('cell'):
        L = meta.at[cid, 'cycle_life']
        x = g.cycle.values / L
        curves[cid] = np.interp(grid, x, rolling_med(g.QD.values, 5) / 1.1, left=np.nan, right=np.nan)
        row = {'cell': cid}
        for lo in [0, .2, .4, .6, .8]:
            mk = (x >= lo) & (x < lo + .2)
            if mk.sum() > 5:
                row[f'fade_{int(lo * 100)}_{int(lo * 100) + 20}'] = -np.polyfit(g.cycle.values[mk], g.QD.values[mk], 1)[0] * 1e4
        gg = g[g.cycle <= L]
        cyc, q = gg.cycle.values[::3], rolling_med(gg.QD.values, 9)[::3]
        row['knee'] = knee_point(cyc, q) if len(cyc) > 30 else np.nan
        rows.append(row)
    D = pd.DataFrame(rows).set_index('cell')
    D['knee_ratio'] = D['knee'] / meta.loc[D.index, 'cycle_life']
    return D, grid, curves
