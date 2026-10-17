"""대시보드 접속 비밀번호 (클라우드 배포용). 비밀번호 자체는 어디에도 저장하지 않고, 솔트를 넣은 해시만 둔다.

비밀번호 해시 만들기 (터미널에서, 입력한 비밀번호는 화면에 보이지 않는다):

    python -m src.auth

나온 세 줄을 Streamlit Community Cloud 앱 설정의 Secrets(또는 로컬 .streamlit/secrets.toml, 커밋하지 않음)에 넣는다:

    [auth]
    salt = "..."
    password_pbkdf2 = "..."
    iterations = 600000

동작
- Secrets에 [auth]가 있으면 비밀번호를 맞혀야 화면이 열린다 (세션마다, 비교는 상수 시간 hmac.compare_digest)
- 틀릴 때마다 대기 시간이 늘고(1·2·4·8·16초), 한 세션에서 5번 틀리면 그 세션은 잠긴다.
  새로 고침하면 새 세션이라 횟수가 다시 시작된다 → 무차별 대입을 늦출 뿐 막지는 못하므로 긴 비밀번호와 아래 비공개 앱 설정이 필요하다
- 클라우드 모드(app_data로 실행)인데 [auth]가 없으면 열지 않는다 (설정을 빠뜨려 공개되는 것을 막는 기본 거부)
  일부러 공개하려면 저장소의 deploy.toml에 [access] public = true 를 커밋하거나(팀 결정 2026-10-15: 공개),
  Secrets에 [auth] disabled = true 를 넣는다
- 이 비밀번호는 1차 방어선이다. Streamlit Community Cloud의 '비공개 앱 + 볼 수 있는 사람 이메일 지정'을 함께 쓴다
"""

import getpass
import hashlib
import hmac
import secrets as pysecrets
import time
import tomllib
from pathlib import Path

DEPLOY_FILE = Path(__file__).resolve().parent.parent / 'deploy.toml'

ITERATIONS = 600_000      # OWASP 권장 PBKDF2-HMAC-SHA256 반복 수 (2023~)
MAX_FAILS = 5


def hash_password(password, salt=None, iterations=ITERATIONS):
    salt = salt or pysecrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), bytes.fromhex(salt), iterations).hex()
    return salt, digest, iterations


def verify(password, salt, expected, iterations=ITERATIONS):
    _, digest, _ = hash_password(password, salt, iterations)
    return hmac.compare_digest(digest, expected)


def public_by_decision(path=DEPLOY_FILE):
    """저장소의 deploy.toml에 [access] public = true가 명시돼 있으면 공개 (팀 결정을 커밋으로 남긴다)"""
    try:
        return bool(tomllib.loads(Path(path).read_text(encoding='utf-8')).get('access', {}).get('public', False))
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return False


def require_login(st, cloud_mode):
    """비밀번호가 맞을 때까지 화면을 막는다. 통과하면 None, 막으면 st.stop()"""
    try:
        conf = st.secrets.get('auth') if 'auth' in st.secrets else None
    except Exception:          # Secrets 파일이 아예 없을 때
        conf = None
    if conf is None:
        if cloud_mode and public_by_decision():
            return             # 팀이 공개로 정함 (deploy.toml)
        if cloud_mode:
            st.error('접속 비밀번호가 설정되지 않아 화면을 열 수 없습니다. 관리자에게 문의하세요.')
            st.stop()
        return                 # 로컬 개발 모드: 내 컴퓨터 안에서만 열리므로 비밀번호 없이
    if conf.get('disabled', False):
        return
    if st.session_state.get('auth_ok'):
        return
    fails = st.session_state.get('auth_fails', 0)
    if fails >= MAX_FAILS:
        st.error('비밀번호를 여러 번 틀려 이 세션이 잠겼습니다. 잠시 뒤 다시 시도하세요.')
        st.stop()
    st.title('연주 설비 조기경보')
    with st.form('login'):
        pw = st.text_input('접속 비밀번호', type='password')
        ok = st.form_submit_button('들어가기')
    if ok:
        if verify(pw, conf['salt'], conf['password_pbkdf2'], int(conf.get('iterations', ITERATIONS))):
            st.session_state['auth_ok'] = True
            st.session_state['auth_fails'] = 0
            st.rerun()
        st.session_state['auth_fails'] = fails + 1
        time.sleep(min(2 ** fails, 16))     # 틀릴수록 오래 기다리게 해 무차별 대입을 늦춘다
        st.error(f'비밀번호가 맞지 않습니다 ({fails + 1}/{MAX_FAILS}).')
    st.stop()


if __name__ == '__main__':
    pw = getpass.getpass('대시보드 접속 비밀번호 (12자 이상 권장): ')
    if len(pw) < 12:
        print('경고: 12자 이상을 권장합니다.')
    if pw != getpass.getpass('한 번 더: '):
        raise SystemExit('두 비밀번호가 다릅니다.')
    salt, digest, it = hash_password(pw)
    print('\n아래를 Secrets에 붙여 넣으세요 (비밀번호 자체는 어디에도 저장하지 않습니다):\n')
    print(f'[auth]\nsalt = "{salt}"\npassword_pbkdf2 = "{digest}"\niterations = {it}')
