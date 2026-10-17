from pathlib import Path
import pandas as pd

# ============================================================
# 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"


# ============================================================
# 파일 찾기
# ============================================================


def find_file(keyword):

    matched_files = [file for file in DATA_DIR.iterdir() if keyword in file.name]

    if len(matched_files) == 0:
        raise FileNotFoundError(
            f"[ERROR] '{keyword}'가 포함된 파일을 찾지 못했습니다.\n"
            f"확인 경로: {DATA_DIR}"
        )

    if len(matched_files) > 1:
        print(f"[WARNING] '{keyword}'에 해당하는 파일이 여러 개 있습니다.")

        for file in matched_files:
            print(" -", file.name)

        print(f"첫 번째 파일을 사용합니다: {matched_files[0].name}")

    return matched_files[0]


# ============================================================
# 데이터 불러오기
# ============================================================


def load_data(path):

    path = Path(path)

    ext = path.suffix.lower()

    if ext == ".csv":

        try:
            df = pd.read_csv(path, low_memory=False)

        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949", low_memory=False)

    elif ext in [".xlsx", ".xls"]:

        df = pd.read_excel(path)

    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    return df
