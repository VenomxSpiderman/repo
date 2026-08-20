import base64
import datetime
import json
import logging
import os
import re
import threading
import time
import bcrypt
import jwt
import streamlit as st
import streamlit.components.v1 as components
import yaml
from dotenv import load_dotenv
from src.storage import get_kv

logger = logging.getLogger('gym_assister.auth')
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'), override=True)
AUTH_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'config', 'auth.yaml')
_AUTH_LOCK = threading.Lock()

def sanitize_auth_username(username: str) -> str:
    if not username:
        return 'default'
    sanitized = re.sub('[^a-zA-Z0-9_-]', '_', str(username).strip())
    return sanitized or 'default'

def get_auth_provider() -> str:
    return 'local'

def _set_auth_cookie(cookie_name: str, username: str, cookie_key: str, expiry_days: float = 30.0):
    exp_ts = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=expiry_days)).timestamp()
    token = jwt.encode({'username': username, 'exp_date': exp_ts}, cookie_key, algorithm='HS256')
    exp_gmt = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=expiry_days)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    components.html(
        f"""
        <script>
            try {{ parent.document.cookie = "{cookie_name}={token}; expires={exp_gmt}; path=/; SameSite=Lax"; }} catch (e) {{}}
            try {{ document.cookie = "{cookie_name}={token}; expires={exp_gmt}; path=/; SameSite=Lax"; }} catch (e) {{}}
        </script>
        """,
        height=0,
        width=0,
    )

def _delete_auth_cookie(cookie_name: str):
    components.html(
        f"""
        <script>
            try {{ parent.document.cookie = "{cookie_name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax"; }} catch (e) {{}}
            try {{ document.cookie = "{cookie_name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax"; }} catch (e) {{}}
        </script>
        """,
        height=0,
        width=0,
    )

def load_auth_config() -> dict:
    kv = get_kv()
    if kv.is_configured():
        raw = kv.get('gym:auth:config')
        if raw:
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(data, dict) and 'credentials' in data:
                    return data
            except Exception as e:
                logger.error('Failed to parse KV auth config: %s', e)
    if os.path.exists(AUTH_CONFIG_PATH):
        with open(AUTH_CONFIG_PATH, 'r', encoding='utf-8') as f:
            local_config = yaml.safe_load(f) or {}
            if kv.is_configured() and local_config:
                kv.set('gym:auth:config', json.dumps(local_config))
            return local_config
    default_config = {
        'credentials': {'usernames': {}},
        'cookie': {
            'name': 'gym_assister_auth',
            'key': 'gym_assister_secret_key_change_me',
            'expiry_days': 30,
        },
    }
    if kv.is_configured():
        kv.set('gym:auth:config', json.dumps(default_config))
    return default_config

def save_auth_config(auth_config: dict):
    kv = get_kv()
    if kv.is_configured():
        kv.set('gym:auth:config', json.dumps(auth_config))
    if os.path.exists(AUTH_CONFIG_PATH):
        try:
            with open(AUTH_CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.safe_dump(auth_config, f, default_flow_style=False)
        except Exception:
            pass

def register_new_user(username, first_name, last_name, email, password, auth_config_path=None):
    safe_user = sanitize_auth_username(username).lower()
    if not safe_user or safe_user == 'default':
        return False, 'Please enter a valid username (letters, numbers, underscores).'
    if not first_name.strip():
        return False, 'Please enter your first name.'
    clean_email = email.strip() if email else ''
    if clean_email and '@' not in clean_email:
        return False, 'Please enter a valid email address, or leave it blank.'
    if not password or len(password) < 8:
        return False, 'Password must be at least 8 characters.'
    with _AUTH_LOCK:
        auth_config = load_auth_config()
        if 'credentials' not in auth_config:
            auth_config['credentials'] = {}
        if 'usernames' not in auth_config['credentials']:
            auth_config['credentials']['usernames'] = {}
        if safe_user in auth_config['credentials']['usernames']:
            return False, f"Username '{safe_user}' already exists. Please choose another."
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        auth_config['credentials']['usernames'][safe_user] = {
            'email': clean_email,
            'first_name': first_name.strip(),
            'last_name': last_name.strip() if last_name else '',
            'logged_in': False,
            'password': hashed,
        }
        save_auth_config(auth_config)
    logger.info("New user '%s' registered successfully", safe_user)
    return True, f"Account '{safe_user}' created successfully! You can now sign in."

def render_sign_in_header():
    logo_path = os.path.join(_PROJECT_ROOT, 'assets', 'logo.jpg')
    img_tag = ''
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as _f:
            img_b64 = base64.b64encode(_f.read()).decode()
        img_tag = f'<img class="login-logo" src="data:image/jpeg;base64,{img_b64}" />'
    st.markdown(
        f'<div class="login-header-wrapper">{img_tag}<div class="login-title">Gym Assister</div><div class="login-subtitle">AI-Powered Training &amp; Diet Tracking</div></div>',
        unsafe_allow_html=True,
    )

def render_local_auth(auth_config_path: str = AUTH_CONFIG_PATH):
    import streamlit_authenticator as stauth
    auth_config = load_auth_config()
    usernames_dict = auth_config.get('credentials', {}).get('usernames', {})
    cookie_name = auth_config.get('cookie', {}).get('name', 'gym_assister_auth')
    cookie_key = auth_config.get('cookie', {}).get('key', 'gym_assister_secret_key_change_me')
    cookie_days = float(auth_config.get('cookie', {}).get('expiry_days', 30.0))

    authenticator = None
    try:
        authenticator = stauth.Authenticate(
            auth_config['credentials'],
            cookie_name,
            cookie_key,
            cookie_days,
        )
    except Exception as e:
        logger.debug('Authenticator init warning: %s', e)

    if not st.session_state.get('authentication_status') and not st.session_state.get('logout'):
        raw_token = None
        if hasattr(st, 'query_params') and st.query_params.get('auth'):
            raw_token = st.query_params.get('auth')
        if not raw_token and hasattr(st, 'context') and hasattr(st.context, 'cookies'):
            raw_token = st.context.cookies.get(cookie_name)
        if raw_token:
            try:
                decoded = jwt.decode(raw_token, cookie_key, algorithms=['HS256'])
                if decoded and isinstance(decoded, dict):
                    uname = decoded.get('username')
                    exp = decoded.get('exp_date', 0)
                    if uname and uname in usernames_dict and exp > datetime.datetime.now(datetime.timezone.utc).timestamp():
                        u_entry = usernames_dict[uname]
                        st.session_state['authentication_status'] = True
                        st.session_state['username'] = uname
                        st.session_state['name'] = f"{u_entry.get('first_name', '')} {u_entry.get('last_name', '')}".strip() or uname
                        st.session_state['auth_username'] = sanitize_auth_username(uname)
                        st.session_state['logout'] = False
                        if hasattr(st, 'query_params'):
                            st.query_params['auth'] = raw_token
            except Exception as e:
                logger.debug('Auto-login decode check: %s', e)

    if st.session_state.get('authentication_status') is True:
        raw_username = st.session_state.get('username', 'default')
        auth_username = sanitize_auth_username(raw_username)
        st.session_state['auth_username'] = auth_username
        if st.session_state.get('_auth_logged_user') != auth_username:
            st.session_state['_auth_logged_user'] = auth_username
            logger.info("User '%s' authenticated successfully", auth_username)
        return authenticator

    st.markdown('''<style>
    .login-header-wrapper {
        text-align: center;
        margin: 2.5rem 0 1.2rem 0;
    }
    .login-logo {
        width: 85px;
        height: 85px;
        object-fit: cover;
        border-radius: 18px;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(139, 92, 246, 0.35);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .login-title {
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .login-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        background: rgba(255, 255, 255, 0.025) !important;
        box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.45) !important;
        padding: 0.5rem 0.2rem !important;
    }
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    div[data-testid="stForm"] button[kind="primary"],
    div[data-testid="stForm"] button[kind="secondary"] {
        width: 100% !important;
        border-radius: 9px !important;
        padding: 0.55rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35) !important;
        margin-top: 0.5rem !important;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    </style>''', unsafe_allow_html=True)
    _, center_col, _ = st.columns([1.1, 1, 1.1])
    with center_col:
        render_sign_in_header()
        with st.container(border=True):
            tab_login, tab_register = st.tabs(['Sign In', 'Create Account'])
            with tab_login:
                with st.form('signin_form', clear_on_submit=False):
                    login_user = st.text_input('Username', key='login_user_in', placeholder='Enter username').strip().lower()
                    login_pwd = st.text_input('Password', type='password', key='login_pwd_in', placeholder='Enter password')
                    submit_login = st.form_submit_button('Sign In', type='primary', use_container_width=True)
                    if submit_login:
                        if not login_user or not login_pwd:
                            st.warning('Please enter both username and password.')
                        elif login_user not in usernames_dict:
                            st.error('Invalid username or password.')
                        else:
                            user_entry = usernames_dict[login_user]
                            hashed_pwd = user_entry.get('password', '')
                            try:
                                is_valid = bcrypt.checkpw(login_pwd.encode('utf-8'), hashed_pwd.encode('utf-8'))
                            except Exception:
                                is_valid = False
                            if is_valid:
                                exp_ts = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=cookie_days)).timestamp()
                                token = jwt.encode({'username': login_user, 'exp_date': exp_ts}, cookie_key, algorithm='HS256')
                                if hasattr(st, 'query_params'):
                                    st.query_params['auth'] = token
                                _set_auth_cookie(cookie_name, login_user, cookie_key, cookie_days)
                                st.session_state['authentication_status'] = True
                                st.session_state['username'] = login_user
                                st.session_state['name'] = f"{user_entry.get('first_name', '')} {user_entry.get('last_name', '')}".strip() or login_user
                                st.session_state['auth_username'] = sanitize_auth_username(login_user)
                                st.session_state['logout'] = False
                                st.rerun()
                            else:
                                st.error('Invalid username or password.')
            with tab_register:
                with st.form('register_form', clear_on_submit=False):
                    st.markdown('##### New User Registration')
                    reg_username = st.text_input('Username *', key='reg_user_in', placeholder='e.g. fitness_is_my_passion')
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        reg_first = st.text_input('First Name *', key='reg_first_in', placeholder='John')
                    with rcol2:
                        reg_last = st.text_input('Last Name', key='reg_last_in', placeholder='Doe')
                    reg_email = st.text_input('Email', key='reg_email_in', placeholder='johndoe@yahoo.com')
                    reg_pwd = st.text_input('Password *', type='password', key='reg_pwd_in')
                    reg_pwd_confirm = st.text_input('Confirm Password *', type='password', key='reg_pwd_confirm_in')
                    submit_reg = st.form_submit_button('Create Account', type='primary', use_container_width=True)
                    if submit_reg:
                        if not reg_username.strip():
                            st.warning('Please enter a username.')
                        elif not reg_first.strip():
                            st.warning('Please enter your first name.')
                        elif not reg_pwd:
                            st.warning('Please enter a password.')
                        elif len(reg_pwd) < 8:
                            st.warning('Password must be at least 8 characters.')
                        elif reg_pwd != reg_pwd_confirm:
                            st.warning('Passwords do not match. Please re-enter your password.')
                        else:
                            ok, msg = register_new_user(reg_username, reg_first, reg_last, reg_email, reg_pwd)
                            if ok:
                                st.success(msg)
                            else:
                                st.warning(msg)
    if not st.session_state.get('authentication_status'):
        st.stop()
    raw_username = st.session_state.get('username', 'default')
    auth_username = sanitize_auth_username(raw_username)
    st.session_state['auth_username'] = auth_username
    if st.session_state.get('_auth_logged_user') != auth_username:
        st.session_state['_auth_logged_user'] = auth_username
        logger.info("User '%s' authenticated successfully", auth_username)
    return authenticator

def authenticate_user():
    return render_local_auth()

def render_logout(authenticator=None):
    auth_name = st.session_state.get('name', st.session_state.get('auth_username', 'User'))
    st.caption(f'Logged in as **{auth_name}**')
    if authenticator and hasattr(authenticator, 'logout'):
        authenticator.logout('Logout', 'sidebar')
        if st.session_state.get('logout'):
            if hasattr(st, 'query_params'):
                st.query_params.clear()
            _delete_auth_cookie('gym_assister_auth')
    else:
        if st.button('Logout', key='app_sidebar_logout_btn', use_container_width=True):
            if hasattr(st, 'query_params'):
                st.query_params.clear()
            st.session_state['authentication_status'] = None
            st.session_state['username'] = None
            st.session_state['name'] = None
            st.session_state['logout'] = True
            st.session_state.pop('_auth_logged_user', None)
            _delete_auth_cookie('gym_assister_auth')
            st.rerun()
