import base64
import hashlib
import json
import logging
import os
import requests
import streamlit as st
from cryptography.fernet import Fernet
from dotenv import load_dotenv

logger = logging.getLogger('gym_assister.storage')
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
DATA_DIR = ''

def ensure_data_dir(username=None):
    pass


class UpstashKV:
    def __init__(self, url=None, token=None):
        self.url = (url or os.getenv('UPSTASH_REDIS_REST_URL') or os.getenv('KV_REST_API_URL') or '').rstrip('/')
        self.token = token or os.getenv('UPSTASH_REDIS_REST_TOKEN') or os.getenv('KV_REST_API_TOKEN') or ''

    def is_configured(self) -> bool:
        return bool(self.url and self.token)

    def execute(self, *command):
        if not self.is_configured():
            return None
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }
        try:
            resp = requests.post(self.url, headers=headers, json=list(command), timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('result')
            logger.error('Upstash KV request failed (status %s): %s', resp.status_code, resp.text)
            return None
        except Exception as e:
            logger.error('Upstash KV connection error: %s', e)
            return None

    def get(self, key):
        return self.execute('GET', str(key))

    def set(self, key, value):
        return self.execute('SET', str(key), str(value))

    def delete(self, *keys):
        return self.execute('DEL', *[str(k) for k in keys])

    def sadd(self, key, *members):
        return self.execute('SADD', str(key), *[str(m) for m in members])

    def srem(self, key, *members):
        return self.execute('SREM', str(key), *[str(m) for m in members])

    def smembers(self, key):
        res = self.execute('SMEMBERS', str(key))
        if isinstance(res, list):
            return set(res)
        return set()

_kv_instance = None

def get_kv() -> UpstashKV:
    global _kv_instance
    if _kv_instance is None:
        _kv_instance = UpstashKV()
    return _kv_instance

def _get_user_cipher(username=None):
    user = username or st.session_state.get('auth_username', 'default')
    key_material = hashlib.sha256(f'gym_assister_salt_{user}'.encode('utf-8')).digest()
    fernet_key = base64.urlsafe_b64encode(key_material)
    return Fernet(fernet_key)

def load_user_api_keys(username=None):
    user = username or st.session_state.get('auth_username', 'default')
    kv = get_kv()
    raw = kv.get(f'gym:user:{user}:api_keys')
    if not raw:
        return {}
    try:
        cipher = _get_user_cipher(username)
        decrypted_bytes = cipher.decrypt(raw.encode('utf-8'))
        return json.loads(decrypted_bytes.decode('utf-8'))
    except Exception as e:
        try:
            return json.loads(raw)
        except Exception:
            logger.warning('Could not decrypt user API keys: %s', e)
            return {}

def save_user_api_keys(keys, username=None):
    user = username or st.session_state.get('auth_username', 'default')
    kv = get_kv()
    try:
        cipher = _get_user_cipher(username)
        payload = json.dumps(keys or {}).encode('utf-8')
        encrypted_bytes = cipher.encrypt(payload)
        kv.set(f'gym:user:{user}:api_keys', encrypted_bytes.decode('utf-8'))
    except Exception as e:
        logger.error('Failed to save user API keys: %s', e)

def list_profiles(username=None):
    user = username or st.session_state.get('auth_username', 'default')
    kv = get_kv()
    profiles = kv.smembers(f'gym:user:{user}:profiles')
    if not profiles:
        return []
    return sorted(list(profiles))

def get_profile_filepath(profile_name, username=None):
    safe_name = ''.join((c for c in profile_name if c.isalnum() or c in (' ', '_', '-'))).strip()
    return (f'{safe_name}.json' if safe_name else None, safe_name)


def _default_profile(safe_name):
    return {'profile_name': safe_name, 'demographics': None, 'chat_history': [], 'approved_schema': None, 'weekly_logs': {}}

def load_profile_data(profile_name, username=None):
    safe_name = ''.join((c for c in profile_name if c.isalnum() or c in (' ', '_', '-'))).strip()
    if not safe_name:
        return _default_profile('default')
    user = username or st.session_state.get('auth_username', 'default')
    kv = get_kv()
    raw = kv.get(f'gym:user:{user}:profile:{safe_name}')
    if not raw:
        return _default_profile(safe_name)
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if 'weekly_logs' not in data:
            data['weekly_logs'] = {}
        data.pop('api_keys', None)
        if 'chat_history' in data and isinstance(data['chat_history'], list):
            for msg in data['chat_history']:
                if isinstance(msg, dict) and 'content' in msg:
                    c = msg['content']
                    if not isinstance(c, str):
                        def _flat(val):
                            if isinstance(val, str):
                                return val
                            if isinstance(val, list):
                                return ''.join((_flat(x) for x in val))
                            if isinstance(val, dict):
                                return str(val.get('text', str(val)))
                            return str(val or '')
                        msg['content'] = _flat(c)
        return data
    except Exception as e:
        logger.warning("Corrupt data for profile '%s': %s", safe_name, e)
        return _default_profile(safe_name)

def save_profile_data(profile_name, data, username=None):
    safe_name = ''.join((c for c in profile_name if c.isalnum() or c in (' ', '_', '-'))).strip()
    if not safe_name:
        return
    user = username or st.session_state.get('auth_username', 'default')
    kv = get_kv()
    clean_data = dict(data)
    clean_data.pop('api_keys', None)
    kv.sadd(f'gym:user:{user}:profiles', safe_name)
    kv.set(f'gym:user:{user}:profile:{safe_name}', json.dumps(clean_data))

def rename_profile(old_name, new_name, username=None):
    safe_old = ''.join((c for c in old_name if c.isalnum() or c in (' ', '_', '-'))).strip()
    safe_new = ''.join((c for c in new_name if c.isalnum() or c in (' ', '_', '-'))).strip()
    if not safe_new or safe_new == safe_old:
        return (False, 'Please enter a different valid profile name.')
    user = username or st.session_state.get('auth_username', 'default')
    kv = get_kv()
    existing = kv.smembers(f'gym:user:{user}:profiles')
    if safe_new in existing:
        return (False, f"Profile '{safe_new}' already exists. Pick a different name.")
    if safe_old in existing:
        data = load_profile_data(safe_old, username)
        data['profile_name'] = safe_new
        save_profile_data(safe_new, data, username)
        kv.srem(f'gym:user:{user}:profiles', safe_old)
        kv.delete(f'gym:user:{user}:profile:{safe_old}')
        return (True, safe_new)
    return (False, 'Original profile not found.')
