"""연주 24시간 조기경보 모델 정의와 교차검증 도우미.

analysis/04_model.py(모델 선택·검증 평가)와 analysis/05_predict_test.py(테스트 세트 예측)가 같은 정의를 쓴다.

    from src.model import MODELS, FEATURES, FINAL, fold_features, oof_from_folds

    make = MODELS[FINAL['model']]
    cols = FEATURES[FINAL['features']]
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_SETS

N_FOLDS = 5
SEED = 31

# 비교하는 특징 묶음. 시간을 세는 특징(FEATURE_SETS['elapsed']: 경과 시간, 재가동 후 시간)은 쓰지 않는다(팀 결정 E)
FEATURES = {
    '센서': FEATURE_SETS['sensor'],
    '센서 + 정비 이력': FEATURE_SETS['sensor'] + FEATURE_SETS['maintenance'],
}

# 비교하는 모델. 불균형(양성 약 11%)이라 클래스 가중치를 맞춘다
MODELS = {
    '로지스틱 회귀': lambda: make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
                                       LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)),
    '랜덤포레스트': lambda: make_pipeline(SimpleImputer(add_indicator=True),
                                     RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=5,
                                                            class_weight='balanced', random_state=SEED, n_jobs=-1)),
    '그래디언트 부스팅': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                          l2_regularization=1.0, class_weight='balanced',
                                                          random_state=SEED),
}

# 04_model.py에서 학습 세트 교차검증 AP로 고르고 검증 세트에서 평가한 최종 조합 (2026-10-13)
FINAL = {'model': '랜덤포레스트', 'features': '센서 + 정비 이력'}

# 예측 간격(시간). 중간발표 버전(git 태그 midterm)은 6시간. 개선 1(analysis/09_hourly.py, 2026-10-15)에서
# 1시간 시점으로 학습·예측하는 쪽이 학습 세트 교차검증 AP가 높고 경보가 평균 3.9시간 빨라 1시간으로 바꿨다.
# 04·06은 중간발표 분석(6시간)이고, 05_predict_test.py(서비스 모델)가 이 값을 쓴다.
PRED_STEP_H = 1

# 2단계 경보의 주의 모델: '지금 열화 중인가'를 맞힌다. 06_caution.py에서 학습 세트 교차검증 AP로 고름 (2026-10-13)
CAUTION_FINAL = {'model': '랜덤포레스트', 'features': '센서'}


def oof_predict(make_model, X, y, groups, n_folds=N_FOLDS):
    """시나리오 단위 n겹 교차검증의 겹 밖 예측 확률 (같은 시나리오는 한 겹에만 들어간다).

    보정값이 필요 없는 특징(기준 예제)에 쓴다. 우리 특징은 보정값을 폴드마다 다시 정해야 하므로
    fold_features + oof_from_folds를 쓴다.
    """
    oof = np.zeros(len(y))
    for tr_idx, te_idx in GroupKFold(n_splits=n_folds).split(X, y, groups):
        m = make_model().fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict_proba(X[te_idx])[:, 1]
    return oof


def fold_features(hr, train, n_folds=N_FOLDS):
    """교차검증 폴드마다 그 폴드의 학습 부분만으로 보정값을 다시 정해 특징을 계산한다.

    보정값(부하 기울기·절편, 표준화 기준)과 그 기준이 되는 정상 시나리오 선택(진단 정답 사용)에
    평가 폴드의 정보가 섞이지 않게 하려는 것이다.

    hr     preprocess.build_hourly 결과(또는 apply_params 결과. 보정 열은 다시 계산된다)
    train  학습 세트의 예측 시점 표 (build_dataset 결과에서 SPLIT == 'train', 인덱스 0..n-1)
    반환   [(tr_idx, te_idx, 특징 표), ...] 폴드 수만큼. 특징 표는 train과 같은 행 순서
    """
    from src.features import TARGET_COLS, make_features, normal_scenarios
    from src.preprocess import apply_params, fit_params

    ids = train['SCENARIO_ID'].unique().tolist()
    h_train = hr[hr['SCENARIO_ID'].isin(ids)]
    folds = []
    for tr_idx, te_idx in GroupKFold(n_splits=n_folds).split(train, groups=train['SCENARIO_ID']):
        fit_ids = train['SCENARIO_ID'].iloc[tr_idx].unique().tolist()
        params = fit_params(h_train[h_train['SCENARIO_ID'].isin(fit_ids)], normal_scenarios(fit_ids))
        folds.append((tr_idx, te_idx, make_features(apply_params(h_train, params), train[TARGET_COLS])))
    return folds


def time_fold_features(hr, train, n_blocks=6):
    """시간 순서 교차검증: 시나리오를 시간 순서(번호 순서 = 시작 시각 순서)로 n_blocks개 묶음으로 나누고,
    k번째 폴드는 앞의 묶음 1..k로 학습해 바로 다음 묶음 k+1을 평가한다 (앞 구간 학습 / 뒤 구간 평가).

    첫 묶음은 평가되지 않는다. 반환은 fold_features와 같은 형식에 더해, 평가된 행 표시(mask)를 준다.
    보정값도 폴드마다 학습 부분만으로 다시 정한다.
    """
    from src.features import TARGET_COLS, make_features, normal_scenarios
    from src.preprocess import apply_params, fit_params

    ids = sorted(train['SCENARIO_ID'].unique())
    blocks = np.array_split(np.array(ids), n_blocks)
    h_train = hr[hr['SCENARIO_ID'].isin(ids)]
    folds = []
    for k in range(1, n_blocks):
        fit_ids = [s for b in blocks[:k] for s in b]
        tr_idx = np.where(train['SCENARIO_ID'].isin(fit_ids))[0]
        te_idx = np.where(train['SCENARIO_ID'].isin(blocks[k]))[0]
        params = fit_params(h_train[h_train['SCENARIO_ID'].isin(fit_ids)], normal_scenarios(fit_ids))
        folds.append((tr_idx, te_idx, make_features(apply_params(h_train, params), train[TARGET_COLS])))
    mask = ~train['SCENARIO_ID'].isin(blocks[0]).values
    return folds, mask


def oof_from_folds(make_model, cols, y, folds):
    """fold_features 결과로 겹 밖 예측 확률을 만든다 (모델·특징 묶음을 바꿔도 폴드 특징은 다시 계산하지 않는다)"""
    oof = np.zeros(len(y))
    for tr_idx, te_idx, F in folds:
        X = F[cols].values.astype(float)
        m = make_model().fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict_proba(X[te_idx])[:, 1]
    return oof
