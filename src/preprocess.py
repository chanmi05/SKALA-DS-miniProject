"""데이터 다운로드 · 로딩 · 품질 진단 · 정제.

- 원본 .mat(MATLAB v7.3, HDF5)은 수 GB라 통째 로드하지 않고 h5py로 필요한 필드만 읽는다.
- 파싱 결과(원본 그대로)만 data/processed/dataset.pkl 에 저장하고, 진단·정제는 get_dataset() 호출 때마다
  다시 계산한다 → 정제 기준을 바꿔도 8GB를 다시 받을 필요가 없다.
- 정제 기준은 원논문 공개 코드(Severson et al. 2019)를 참고하되, 이 데이터에서 실제로 관측되는 근거로 결정한다.
  주의 : Kaggle의 Batch 2 파일(2018-02-20)은 원논문의 Batch 2(2017-06-30)와 다른 실험이다.
         따라서 논문의 'Batch 1 ← Batch 2 이어붙이기'는 이 파일에 적용할 수 없고,
         Batch 1 중도 종료 셀 5개에는 논문이 보고한 최종 수명 라벨만 사용한다.

사용 예
    from src.preprocess import build_dataset, get_dataset
    build_dataset()           # 처음 1회 : 다운로드 + 파싱 → data/processed/dataset.pkl
    ds = get_dataset()        # 진단 + 정제가 적용된 데이터
"""
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import h5py

from .utils import PROCESSED_DIR

KAGGLE_HANDLE = 'itshpark/data-driven-prediction-of-battery-cycle'
BATCH_KEYS = {'B1': '2017-05-12', 'B2': '2018-02-20', 'B3': '2018-04-12'}

NOMINAL = 1.1        # 공칭 용량 (Ah), A123 APR18650M1A
EOL_Q = 0.88         # EOL = 공칭의 80%

# ΔQ(V) 계산용으로 Qdlin을 읽어올 사이클 (02_feature_engineering의 사이클 조합 분석까지 커버)
Q_CYCLES = [2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PROFILE_CYCLE = 10   # 충전 전류 프로파일을 읽어올 사이클

# 원논문 공개 코드의 정제 기준
PAPER_B1_DROP = ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']                  # 80% 미도달, 이어진 실험 없음
# 논문 : b1c0~4는 2017-06-30 배치(b2c7,8,9,15,16)에서 이어서 측정 → 최종 수명 = 기존 + 아래 사이클 수
PAPER_CONT = {'b1c0': ('b2c7', 662), 'b1c1': ('b2c8', 981), 'b1c2': ('b2c9', 1060),
              'b1c3': ('b2c15', 208), 'b1c4': ('b2c16', 482)}
PAPER_B3_NOISY = ['b3c2', 'b3c23', 'b3c32', 'b3c37', 'b3c42', 'b3c43']          # noisy channels
NONSTANDARD = ('VarCharge', 'SLOWCYCLE')    # 고정 충전 정책이 아닌 실험 (수명 라벨도 없음)
EOL_TOL = 0.005                             # EOL 도달 판정 허용오차 (Ah) : 실험이 0.88 Ah 근처에서 종료되기 때문

PROCESSED_PATH = os.path.join(PROCESSED_DIR, 'dataset.pkl')   # 파싱 결과(원본) 캐시


# ----------------------------------------------------------------------------- download
def download(data_path=None):
    """kagglehub로 데이터셋 다운로드 후 배치별 .mat 경로 반환."""
    data_path = data_path or os.environ.get('BATTERY_DATA_PATH')
    if data_path is None:
        import kagglehub
        data_path = kagglehub.dataset_download(KAGGLE_HANDLE)
    print('Path to dataset files:', data_path)
    for root, _, files in os.walk(data_path):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print(f'  {os.path.relpath(p, data_path):70s} {os.path.getsize(p) / 1024 ** 3:6.2f} GB')
    out = {}
    for b, key in BATCH_KEYS.items():
        cands = [p for p in glob.glob(os.path.join(data_path, '**', '*'), recursive=True)
                 if key in os.path.basename(p) and p.endswith('.mat') and 'varcharge' not in p]
        assert cands, f'{key} .mat 파일을 찾지 못했습니다. 위 파일 목록을 확인해주세요.'
        out[b] = sorted(cands)[0]
    return out


# ----------------------------------------------------------------------------- h5py loader
def _ref(ds, i):
    return ds[i, 0] if ds.shape[1] == 1 else ds[0, i]


def _n(ds):
    return max(ds.shape)


def _vec(f, ref):
    a = np.array(f[ref])
    if a.dtype == np.uint64 and a.size <= 2:          # MATLAB 빈 배열
        return np.array([])
    return a.astype(float).ravel()


def _str(f, ref):
    return np.array(f[ref]).astype(np.uint16).tobytes().decode('utf-16-le', errors='ignore').strip('\x00 ')


def load_batch(path, tag):
    """한 배치 파일에서 셀 단위 dict 생성 (summary 전체 + 일부 사이클의 Qdlin + 충전 프로파일)."""
    out, vdlin = {}, None
    with h5py.File(path, 'r') as f:
        b = f['batch']
        for i in range(_n(b['summary'])):
            cid = f'{tag.lower()}c{i}'
            cl = _vec(f, _ref(b['cycle_life'], i))
            sg = f[_ref(b['summary'], i)]
            summ = {k: np.array(sg[k]).astype(float).ravel()
                    for k in ['cycle', 'QDischarge', 'QCharge', 'IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime']}
            cg = f[_ref(b['cycles'], i)]
            ncyc = _n(cg['Qdlin'])

            def jidx(c):
                hit = np.where(summ['cycle'] == c)[0]
                j = int(hit[0]) if hit.size else c
                return j if j < ncyc else None

            qdlin = {}
            for c in Q_CYCLES:
                j = jidx(c)
                v = _vec(f, _ref(cg['Qdlin'], j)) if j is not None else np.array([])
                qdlin[c] = v if v.size == 1000 else None
            j = jidx(PROFILE_CYCLE)
            prof = {k: _vec(f, _ref(cg[k], j)) for k in ['I', 't', 'Qc', 'V']} if j is not None else None
            if vdlin is None and 'Vdlin' in b:
                v = _vec(f, _ref(b['Vdlin'], i))
                vdlin = v if v.size == 1000 else None
            out[cid] = dict(batch=tag, cycle_life=float(cl[0]) if cl.size else np.nan,
                            policy=_str(f, _ref(b['policy_readable'], i)),
                            summary=summ, qdlin=qdlin, profile=prof)
    return out, (vdlin if vdlin is not None else np.linspace(3.6, 2.0, 1000))


# ----------------------------------------------------------------------------- frames
POLICY_RE = re.compile(r'([\d.]+)C\((\d+(?:\.\d+)?)%\)-([\d.]+)C')


def parse_policy(s):
    """'5.4C(40%)-3.6C' → (C1=5.4, Q1=40, C2=3.6, avgC = 0→80% SOC 평균 C-rate)"""
    m = POLICY_RE.search(s or '')
    if not m:
        return np.nan, np.nan, np.nan, np.nan
    c1, q1, c2 = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
    hours = 0.8 / c1 if q1 >= 0.8 else q1 / c1 + (0.8 - q1) / c2
    return c1, q1 * 100, c2, 0.8 / hours


def to_frames(raw):
    meta_rows, summ_rows = [], []
    for cid, d in raw.items():
        s = d['summary']
        meta_rows.append(dict(cell=cid, batch=d['batch'], policy=d['policy'], cycle_life=d['cycle_life'],
                              n_cycles=len(s['QDischarge'])))
        summ_rows.append(pd.DataFrame(dict(cell=cid, batch=d['batch'], cycle=s['cycle'], QD=s['QDischarge'],
                                           QC=s['QCharge'], IR=s['IR'], Tavg=s['Tavg'], Tmin=s['Tmin'],
                                           Tmax=s['Tmax'], chargetime=s['chargetime'])))
    return pd.DataFrame(meta_rows).set_index('cell'), pd.concat(summ_rows, ignore_index=True)


def add_policy_columns(meta):
    """정책 문자열 → 수치 분해 + 셀 구조(newstructure) 플래그 + 비표준 실험 플래그."""
    meta = meta.copy()
    meta['structure'] = np.where(meta['policy'].str.contains('newstructure', case=False), 'new', 'old')
    meta['policy_base'] = (meta['policy'].str.replace('-newstructure', '', case=False)
                           .str.replace(r'\(SLOWCYCLE.*', '', regex=True).str.strip())
    meta['nonstandard'] = meta['policy'].apply(lambda s: any(k.lower() in s.lower() for k in NONSTANDARD))
    parsed = meta['policy_base'].apply(parse_policy)
    meta[['C1', 'Q1', 'C2', 'avgC']] = pd.DataFrame(parsed.tolist(), index=meta.index)
    return meta


# ----------------------------------------------------------------------------- diagnose / clean
def rolling_med(x, w=11):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().values


def diagnose(meta_raw, summ_raw, qdlin=None):
    """셀별 품질 지표 : 더미 행, 스파이크, 초기/최종 용량, EOL 도달 여부, 노이즈, 온도 센서 이상."""
    rows = []
    for cid, g in summ_raw.groupby('cell', sort=False):
        qd = g['QD'].values
        valid = qd > 0
        q = qd[valid]
        spike = (np.abs(q - rolling_med(q)) > 0.02) | (q > 1.3) | (q < 0.5)
        q_ok = q[~spike]
        sm = rolling_med(q_ok, 5)
        ge = g[valid][~spike]
        early = ge[(ge['cycle'] >= 2) & (ge['cycle'] <= 100)]
        # 노이즈 : (1) 용량 곡선의 사이클 간 흔들림 (2) 사이클 100 방전 곡선 Q(V)의 2차 차분 크기
        jitter = np.std(np.diff(q_ok)) * 1e3 if len(q_ok) > 3 else np.nan
        qv = (qdlin or {}).get(cid, {}).get(100) if qdlin else None
        qv_rough = np.median(np.abs(np.diff(qv, 2))) * 1e6 if qv is not None else np.nan
        rows.append(dict(
            cell=cid, zero_rows=int((~valid).sum()), spikes=int(spike.sum()),
            init_QD=np.median(q_ok[1:10]) if len(q_ok) > 10 else np.nan,
            final_QD=np.median(q_ok[-5:]) if len(q_ok) else np.nan,
            min_QD=np.nanmin(sm[10:]) if len(sm) > 10 else np.nan,
            reached_EOL=bool(len(sm) > 10 and np.nanmin(sm[10:]) <= EOL_Q + EOL_TOL),
            jitter=jitter, qv_rough=qv_rough,
            IR_zero=int((ge['IR'] <= 0).sum()),
            T_range_early=(early['Tmax'] - early['Tmin']).median(),
            Tavg_early=early['Tavg'].median()))
    d = pd.DataFrame(rows).set_index('cell')
    out = add_policy_columns(meta_raw).join(d)
    for col in ['T_range_early', 'Tavg_early', 'jitter', 'qv_rough']:
        grp = out.groupby('batch')[col]
        med = grp.transform('median')
        mad = grp.transform(lambda s: (s - s.median()).abs().median()) * 1.4826 + 1e-9
        out[f'{col}_z'] = (out[col] - med) / mad
    out['temp_suspect'] = (out['T_range_early_z'] < -4) | (out['Tavg_early_z'].abs() > 4) | out['T_range_early'].isna()
    return out


def clean_summary(summ_raw):
    """더미 행(QD=0) · QD 스파이크 제거, IR=0(측정 누락)과 다른 변수의 극단값은 NaN 처리."""
    parts = []
    for _, g in summ_raw.groupby('cell', sort=False):
        g = g[g['QD'] > 0].copy()
        q = g['QD'].values
        bad = (np.abs(q - rolling_med(q)) > 0.02) | (q > 1.3) | (q < 0.5)
        g = g[~bad]
        g.loc[g['IR'] <= 0, 'IR'] = np.nan
        for col in ['IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']:
            x = g[col].values
            m = pd.Series(x).rolling(21, center=True, min_periods=1).median().values
            mad = np.nanmedian(np.abs(x - m)) + 1e-9
            g.loc[np.abs(x - m) > 10 * 1.4826 * mad, col] = np.nan
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def apply_cleaning(meta_diag, summ_clean, drop_b3_noisy=True):
    """정제 규칙 (각 셀에 사유를 남긴다).

    1. 비표준 실험(VarCharge, SLOWCYCLE) 제외
    2. b1c0~4 : EOL 미도달이지만 논문이 이어진 실험으로 확정한 최종 수명 라벨 사용 (곡선은 EOL 전까지만 존재)
    3. 그 외 EOL 미도달 · 수명 라벨 없음 → 제외 (실제 수명을 알 수 없음)
    4. Batch 3 논문 지정 노이즈 채널 제외 (drop_b3_noisy)
    """
    m = meta_diag.copy()
    m['cycle_life_raw'] = m['cycle_life']
    m['label_source'] = 'data'
    m['curve_complete'] = m['reached_EOL']
    reason = pd.Series('', index=m.index)
    reason[m['nonstandard']] = 'non-standard protocol'
    for c, (_, add) in PAPER_CONT.items():
        if c in m.index:
            m.at[c, 'cycle_life'] = m.at[c, 'cycle_life'] + add
            m.at[c, 'label_source'] = 'paper (continued test)'
    unknown = (~m['reached_EOL'] | m['cycle_life'].isna()) & (m['label_source'] == 'data')
    reason[unknown & (reason == '')] = 'censored / no label'
    if drop_b3_noisy:
        reason[m.index.isin(PAPER_B3_NOISY) & (reason == '')] = 'paper: noisy channel'
    m['drop_reason'] = reason
    meta = m[reason == ''].copy()
    summ = summ_clean[summ_clean.cell.isin(meta.index)].sort_values(['cell', 'cycle']).reset_index(drop=True)
    return meta, summ, m


# ----------------------------------------------------------------------------- pipeline
def build_dataset(force=False, data_path=None):
    """다운로드 → h5py 파싱 → data/processed/dataset.pkl (원본 그대로) 저장."""
    if os.path.exists(PROCESSED_PATH) and not force:
        print('이미 파싱된 데이터가 있습니다 →', PROCESSED_PATH)
        return
    files = download(data_path)
    raw, vdlin = {}, None
    for tag, p in files.items():
        print(f'{tag} 로딩 중... ({os.path.getsize(p) / 1024 ** 3:.1f} GB) {os.path.basename(p)}')
        d, v = load_batch(p, tag)
        raw.update(d)
        vdlin = v if vdlin is None else vdlin
        print(f'  → {len(d)} cells')
    meta_raw, summ_raw = to_frames(raw)
    ds = dict(meta_raw=meta_raw, summ_raw=summ_raw, vdlin=vdlin,
              qdlin={c: d['qdlin'] for c, d in raw.items()},
              profile={c: d['profile'] for c, d in raw.items()})
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(PROCESSED_PATH, 'wb') as fp:
        pickle.dump(ds, fp)
    print('저장 :', PROCESSED_PATH, f'({os.path.getsize(PROCESSED_PATH) / 1024 ** 2:.1f} MB)')


def get_dataset(drop_b3_noisy=True):
    """파싱 캐시 로드 → 진단 → 정제. (이전 버전 캐시도 그대로 읽는다)"""
    with open(PROCESSED_PATH, 'rb') as fp:
        ds = pickle.load(fp)
    base = ds['meta_raw'][['batch', 'policy', 'cycle_life', 'n_cycles']].copy()
    summ_raw = ds['summ_raw']
    meta_raw = diagnose(base, summ_raw, ds['qdlin'])
    summ_clean = clean_summary(summ_raw)
    meta, summ, meta_all = apply_cleaning(meta_raw, summ_clean, drop_b3_noisy)
    return dict(meta_raw=meta_all, summ_raw=summ_raw, meta=meta, summ=summ, vdlin=ds['vdlin'],
                qdlin=ds['qdlin'], profile=ds['profile'])
