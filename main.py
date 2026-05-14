"""
========================================================================
 메인 분석 프로그램
========================================================================

데이터 흐름:
    1) HY202103 폴더에서 모든 LMZC/LMZO XML 자동 탐색
    2) 다이별로 ER, IL, V_π 추출 — 멀티코어 병렬 처리
    3) Outlier 검출 (물리 한계 + Hampel filter)
    4) 실행 시각 폴더 만들고 CSV 저장
    5) 웨이퍼맵 / 1D분포 / 1D+MAD / 신뢰도맵 — 4개 동시 생성

VSCode ▶ (F5) 로 그대로 실행 가능. 절대경로 사용.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from xml_loader import find_all_xmls, load_die
from extract_er import extract_er
from extract_il import extract_il
from extract_vpi import extract_vpi
from outlier_detect import mark_outliers
from csv_export import make_run_dir, export_csv
import wafer_map
import plot_1d
import plot_1d_mad
import trust_map


DATA_ROOT = '/Users/gimhanseo/Desktop/공프/HY202103'


def process_die(xml_path):
    """단일 XML → 한 행 dict (ER, IL, Vpi 포함)."""
    die = load_die(xml_path)
    if die is None:
        return None
    er = extract_er(die)
    il = extract_il(die)
    vpi_info = extract_vpi(die)
    return {
        'Wafer':    die['wafer'],
        'Band':     die['band'],
        'Row':      die['row'],
        'Col':      die['col'],
        'Width_nm': die['width_nm'],
        'ER_dB':    er,
        'IL_dB':    il,
        'Vpi_V':    vpi_info['vpi_V'],
        'FSR_nm':           vpi_info['fsr_nm'],
        'dlam_dV_pm_per_V': vpi_info['dlam_dV_pm_per_V'],
    }


METRICS = [
    ('ER_dB', 'Extinction Ratio (dB)'),
    ('IL_dB', 'Insertion Loss (dB)'),
    ('Vpi_V', 'V_pi (V)'),
]

# ── 플롯 작업 래퍼 — metric별로 쪼개서 12개 작업으로 (8코어 풀 활용) ──
def _run_plot(args):
    """(plot_type, col, label, df, run_dir) → 개별 그래프 1장 생성."""
    import os
    plot_type, col, label, df, run_dir = args
    if plot_type == 'wafer':
        import wafer_map as _m
        _m.plot_wafer_map(df, col, label, os.path.join(run_dir, f'wafer_map_{col}.png'))
    elif plot_type == '1d':
        import plot_1d as _m
        _m.plot_1d(df, col, label, os.path.join(run_dir, f'1d_{col}.png'))
    elif plot_type == '1d_mad':
        import plot_1d_mad as _m
        _m.plot_1d_mad(df, col, label, os.path.join(run_dir, f'1d_mad_{col}.png'))
    elif plot_type == 'trust':
        import trust_map as _m
        _m.plot_trust_map(df, col, label, os.path.join(run_dir, f'trust_map_{col}.png'))


def main():
    print('=' * 60)
    print(' MZM 4-wafer 분석 시작')
    print('=' * 60)

    # 1) XML 수집
    xmls = find_all_xmls(DATA_ROOT)
    print(f'\n[1/6] 발견된 다이 XML: {len(xmls)}개')

    # 2) 다이별 추출 — 멀티코어 병렬
    print('[2/6] ER, IL, V_π 추출 중... (멀티코어)')
    with multiprocessing.Pool() as pool:
        results = pool.map(process_die, xmls)
    rows = [r for r in results if r is not None]
    df = pd.DataFrame(rows).sort_values(['Band', 'Wafer', 'Row', 'Col'])
    df = df.reset_index(drop=True)
    print(f'        → {len(df)}개 다이 처리 완료')

    # 3) Outlier 검출
    print('[3/6] Outlier 검출 (물리 바운드 + Hampel)...')
    df = mark_outliers(df)
    n_trusted = int(df['is_trusted'].sum())
    print(f'        → 신뢰 다이 {n_trusted}/{len(df)}개')
    for col in ['ER_dB', 'IL_dB', 'Vpi_V']:
        n_out = int(df[f'is_outlier_{col}'].sum())
        print(f'         {col}: outlier {n_out}개')

    # 4) 실행 폴더 + CSV
    print('[4/6] 결과 폴더 생성 + CSV 저장...')
    run_dir = make_run_dir()
    print(f'        → {run_dir}')
    export_csv(df, run_dir)

    # 5~7) 플롯 12개 동시 생성 — 8코어 풀 활용
    print('[5/7] 플롯 생성 (4종 × 3 metric = 12작업, 8코어 병렬)...')
    tasks = [
        ('wafer',  col, label, df, run_dir)
        for col, label in METRICS
    ] + [
        ('1d',     col, label, df, run_dir)
        for col, label in METRICS
    ] + [
        ('1d_mad', col, label, df, run_dir)
        for col, label in METRICS
    ] + [
        ('trust',  col, label, df, run_dir)
        for col, label in METRICS
    ]
    with ProcessPoolExecutor(max_workers=None) as ex:
        futures = [ex.submit(_run_plot, t) for t in tasks]
        for f in futures:
            f.result()
    print('        → 모든 플롯 완료')

    print('\n' + '=' * 60)
    print(' 완료!  결과 위치:')
    print(f'   {run_dir}')
    print('=' * 60)

    # 요약 통계 (신뢰 데이터만)
    trusted = df[df['is_trusted']]
    print('\n[중앙값 요약 (신뢰 데이터만)]')
    summary = trusted.groupby(['Band', 'Wafer']).agg(
        n=('Row', 'count'),
        ER=('ER_dB', 'median'),
        IL=('IL_dB', 'median'),
        Vpi=('Vpi_V', 'median'),
    ).round(2)
    print(summary)


if __name__ == '__main__':
    main()
