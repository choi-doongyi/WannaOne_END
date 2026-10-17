"""그림 공통 설정: 한글 글꼴(macOS·Windows·Linux)과 그림 저장.

    from src.plotting import setup_korean_font, save_figure

    setup_korean_font()                       # 스크립트 맨 앞에서 한 번
    save_figure(fig, '04_eda_리드타임.png')     # figures/에 저장

- 글꼴: 운영체제마다 기본 한글 글꼴이 달라서, 설치된 글꼴 중 먼저 찾은 것을 쓴다.
  어느 것도 없으면 경고를 내고, 나눔고딕 설치를 안내한다.
- 파일 이름: 한글 파일 이름을 NFC로 맞춰 저장한다. macOS에서 자모가 분리된(NFD) 이름으로 저장되면
  Windows에서 "ㅎㅏㄴㄱㅡㄹ"처럼 보이기 때문이다.
- PNG는 글자를 그림으로 그려 넣으므로, 저장된 그림은 어느 운영체제에서 열어도 깨지지 않는다.
"""

import unicodedata
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

FIG_DIR = Path(__file__).resolve().parent.parent / 'figures'

# 앞에서부터 찾는다: macOS → Windows → 공통(나눔고딕) → Linux
KOREAN_FONTS = ['AppleGothic', 'Apple SD Gothic Neo', 'Malgun Gothic', 'NanumGothic', 'Nanum Gothic',
                'Noto Sans CJK KR', 'Noto Sans KR', 'UnDotum']


def setup_korean_font():
    """설치된 한글 글꼴을 찾아 matplotlib 기본 글꼴로 정한다. 찾은 글꼴 이름을 돌려준다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in installed), None)
    if font is None:
        warnings.warn('한글 글꼴을 찾지 못해 그림의 한글이 깨질 수 있다. 나눔고딕을 설치한 뒤 '
                      'matplotlib 글꼴 캐시(~/.matplotlib 또는 %USERPROFILE%\\.matplotlib)를 지우고 다시 실행할 것')
    else:
        plt.rcParams['font.family'] = font
    plt.rcParams['axes.unicode_minus'] = False      # 한글 글꼴에 없는 '−' 기호 대신 '-'
    return font


def save_figure(fig, name, dpi=120):
    """figures/에 그림을 저장한다 (파일 이름은 NFC). 저장한 경로를 돌려준다."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / unicodedata.normalize('NFC', name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 그림 저장: figures/{path.name}')
    return path
