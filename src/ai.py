import datetime
import json
import logging
import os
import re
import time
import requests
import yaml
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.models import ApprovedSchema

logger = logging.getLogger('gym_assister.ai')
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))
PROMPTS_FILE = os.path.join(_PROJECT_ROOT, 'config', 'prompt.yml')

def load_prompts():
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

PROMPTS = load_prompts()
MODEL_NAME = ''
SUPPORTED_PROVIDERS = ['Groq', 'Google Gemini', 'OpenAI', 'Anthropic']

PROVIDER_ENV_KEYS = {
    'Groq': 'GROQ_API_KEY',
    'Google Gemini': 'GEMINI_API_KEY',
    'OpenAI': 'OPENAI_API_KEY',
    'Anthropic': 'ANTHROPIC_API_KEY',
}

def is_local_ai_enabled() -> bool:
    return False

def get_available_providers() -> list:
    return list(SUPPORTED_PROVIDERS)

def get_default_provider() -> str:
    return SUPPORTED_PROVIDERS[0]

def is_valid_chat_model(model_name: str) -> bool:
    if not model_name or not isinstance(model_name, str):
        return False
    lower = model_name.lower()
    excluded_keywords = (
        'vision',
        'embed',
        'whisper',
        'tts',
        'stt',
        'audio',
        'speech',
        'transcribe',
        'dall-e',
        'dalle',
        'imagen',
        'image',
        'moderation',
        'guard',
        'shield',
        'realtime',
        'rerank',
        'bge-',
        'minilm',
    )
    return not any(kw in lower for kw in excluded_keywords)

def validate_and_fetch_models(provider, api_key=None):
    provider_clean = (provider or '').strip()
    try:
        key = (api_key or '').strip()
        if not key:
            return False, [], 'Please enter an API key.'
        if 'Groq' in provider_clean:
            headers = {'Authorization': f'Bearer {key}'}
            resp = requests.get('https://api.groq.com/openai/v1/models', headers=headers, timeout=5)
            if resp.status_code == 200:
                models = [m.get('id') for m in resp.json().get('data', []) if m.get('id') and is_valid_chat_model(m.get('id'))]
                if models:
                    return True, sorted(models), f'API key verified! Loaded {len(models)} model(s).'
            return False, [], f'Invalid API key (HTTP {resp.status_code}).'
        elif 'Gemini' in provider_clean or 'Google' in provider_clean:
            url = 'https://generativelanguage.googleapis.com/v1beta/models'
            headers = {'x-goog-api-key': key}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                raw_models = resp.json().get('models', [])
                models = []
                for m in raw_models:
                    name = m.get('name', '')
                    methods = m.get('supportedGenerationMethods', [])
                    if 'generateContent' in methods and name.startswith('models/'):
                        clean_m = name.replace('models/', '')
                        if is_valid_chat_model(clean_m) and not any(k in clean_m.lower() for k in ('2.5', 'custom', 'tuning')):
                            models.append(clean_m)
                if not models:
                    models = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
                return True, sorted(models), f'API key verified! Loaded {len(models)} model(s).'
            return False, [], f'Invalid API key (HTTP {resp.status_code}).'

        elif 'OpenAI' in provider_clean:
            headers = {'Authorization': f'Bearer {key}'}
            resp = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=5)
            if resp.status_code == 200:
                models = [m.get('id') for m in resp.json().get('data', []) if m.get('id') and any(k in m.get('id') for k in ('gpt', 'o1', 'o3', 'chatgpt')) and is_valid_chat_model(m.get('id'))]
                if models:
                    return True, sorted(models), f'API key verified! Loaded {len(models)} model(s).'
            return False, [], f'Invalid API key (HTTP {resp.status_code}).'
        elif 'Anthropic' in provider_clean:
            headers = {'x-api-key': key, 'anthropic-version': '2023-06-01'}
            resp = requests.get('https://api.anthropic.com/v1/models', headers=headers, timeout=5)
            if resp.status_code == 200:
                models = [m.get('id') for m in resp.json().get('data', []) if m.get('id') and is_valid_chat_model(m.get('id'))]
                if models:
                    return True, sorted(models), f'API key verified! Loaded {len(models)} model(s).'
            return False, [], f'Invalid API key (HTTP {resp.status_code}).'
    except Exception as e:
        logger.debug('Model validation failed for %s: %s', provider, e)
        return False, [], f'Connection error: {str(e)}'
    return False, [], 'Invalid provider or unsupported configuration.'

def fetch_provider_models(provider, api_key=None):
    valid, models, _ = validate_and_fetch_models(provider, api_key)
    return models if valid else []

@st.cache_resource
def get_llm_instance(provider, model_name, api_key=None, temperature=0.2, num_ctx=2048, max_tokens=4096):
    provider_clean = (provider or '').strip().lower()
    if 'groq' in provider_clean:
        key = api_key or os.getenv('GROQ_API_KEY')
        if not key:
            raise ValueError('Groq API key missing. Please enter your GROQ_API_KEY in top-right options.')
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, api_key=key, temperature=temperature, max_tokens=max_tokens)
    elif 'gemini' in provider_clean or 'google' in provider_clean:
        key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not key:
            raise ValueError('Google Gemini API key missing. Please enter your GEMINI_API_KEY in top-right options.')
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=key, temperature=temperature, max_output_tokens=max_tokens)
    elif 'openai' in provider_clean:
        key = api_key or os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError('OpenAI API key missing. Please enter your OPENAI_API_KEY in top-right options.')
        from langchain_openai import ChatOpenAI
        is_reasoning = any(model_name.lower().startswith(p) for p in ('o1', 'o3', 'o4'))
        if is_reasoning:
            return ChatOpenAI(model=model_name, api_key=key, max_completion_tokens=max_tokens)
        return ChatOpenAI(model=model_name, api_key=key, temperature=temperature, max_tokens=max_tokens)
    elif 'anthropic' in provider_clean or 'claude' in provider_clean:
        key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not key:
            raise ValueError('Anthropic API key missing. Please enter your ANTHROPIC_API_KEY in top-right options.')
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=key, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f'Unsupported cloud provider: {provider}')


def clean_and_parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[:-3] if cleaned.endswith('```') else cleaned[3:]
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search('\\{.*\\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def format_error_message(err, provider=None, model_name=None):
    err_str = str(err)
    err_lower = err_str.lower()
    if 'terms' in err_lower or 'model_terms_required' in err_lower:
        return f'Model terms acceptance required for {model_name or "this model"}. Please choose another model in settings.'
    if 'rate_limit' in err_lower or '429' in err_lower or 'tokens per minute' in err_lower or '413' in err_lower or 'tpm' in err_lower or 'quota' in err_lower:
        return f'Rate/token limit exceeded on {provider or "LLM"} ({model_name or "model"}). Please wait a minute or select a higher capacity model.'
    if 'auth' in err_lower or 'api_key' in err_lower or '401' in err_lower or 'unauthorized' in err_lower or 'forbidden' in err_lower or '403' in err_lower or 'invalid api key' in err_lower:
        return f'Authentication error with {provider or "LLM"}. Please verify your API key in LLM settings.'
    if 'connection' in err_lower or 'timeout' in err_lower or 'connect' in err_lower:
        return f'Could not connect to {provider or "LLM"} service. Please check your network or server status.'
    clean_msg = err_str.split(':', 1)[-1].strip() if ':' in err_str else err_str
    return f'Plan generation could not be completed: {clean_msg}'

def extract_text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join((extract_text_content(item) for item in content))
    if isinstance(content, dict):
        return str(content.get('text', str(content)))
    return str(content or '')

def generate_approved_schema(chat_history, provider, model_name, api_key=None):
    llm = get_llm_instance(provider, model_name, api_key=api_key, temperature=0.1, num_ctx=2048)
    strict_system_prompt = PROMPTS.get('schema_system_prompt', "Convert the finalized fitness plan into a strict JSON object. Output ONLY valid JSON, no markdown. Schema: {'workout_days': [{'day': 'String', 'exercises': [{'name': 'String', 'sets': Int, 'reps': Int}]}], 'diet': {'calories': Int, 'protein': Int}}.")
    conversation_str = ''
    truncated_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in truncated_history:
        role = 'User' if msg.get('role') == 'user' else 'Trainer'
        content_str = extract_text_content(msg.get('content', ''))
        conversation_str += f"{role}: {content_str}\n\n"
    human_prompt_tmpl = PROMPTS.get('schema_human_prompt', 'Here is the agreed fitness plan conversation:\n\n{conversation_str}\n\nPlease generate the JSON object now according to the exact schema.')
    messages = [SystemMessage(content=strict_system_prompt.strip()), HumanMessage(content=human_prompt_tmpl.format(conversation_str=conversation_str).strip())]
    retry_prompt = PROMPTS.get('schema_retry_prompt', "Your output failed parsing. Output ONLY valid JSON matching the exact schema: {'workout_days': [{'day': 'String', 'exercises': [{'name': 'String', 'sets': Int, 'reps': Int}]}], 'diet': {'calories': Int, 'protein': Int}}.").strip()
    max_retries = 2
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            content_str = extract_text_content(response.content)
            parsed = clean_and_parse_json(content_str)
            validated = ApprovedSchema.model_validate(parsed)
            logger.info('Schema generated successfully on attempt %d', attempt + 1)
            return validated.model_dump()
        except Exception as e:
            last_error = e
            logger.warning('Schema generation attempt %d failed: %s', attempt + 1, e)
            err_lower = str(e).lower()
            if any((k in err_lower for k in ['rate_limit', '429', '413', 'tpm', 'tokens per minute', 'auth', '401', '403', 'quota'])):
                raise
            if 'response' in locals() and hasattr(response, 'content'):
                messages.append(AIMessage(content=extract_text_content(response.content)))
            messages.append(HumanMessage(content=retry_prompt))
    logger.error('Schema generation failed after %d attempts: %s', max_retries, last_error)
    raise Exception(f'Failed to generate valid JSON schema after {max_retries} attempts: {last_error}')


@st.cache_data(show_spinner=False)
def fetch_cached_ai_summary(active_profile, metrics_json, goal, level, provider, model_name, api_key=None):
    metrics = json.loads(metrics_json)
    llm = get_llm_instance(provider, model_name, api_key=api_key, temperature=0.2, num_ctx=2048)
    sys_prompt = PROMPTS.get('summary_system_prompt', 'You are an elite personal trainer giving a concise progress review.').strip()
    user_prompt_tmpl = PROMPTS.get('summary_user_prompt', '')
    prompt = user_prompt_tmpl.format(
        active_profile=active_profile,
        first_date=metrics['first_date'],
        latest_date=metrics['latest_date'],
        goal=goal,
        level=level,
        adherence_pct=metrics['adherence_pct'],
        completed_exercises=metrics['completed_exercises'],
        total_assigned=metrics['total_assigned'],
        avg_calories=metrics['avg_calories'],
        target_calories=metrics['target_calories'],
        weight_change=metrics['weight_change'],
        initial_weight=metrics['initial_weight'],
        latest_weight=metrics['latest_weight'],
        scheduled_rest_count=metrics['scheduled_rest_count'],
        additional_rest_count=metrics['additional_rest_count'],
    ).strip()
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content=prompt)]
    try:
        response = llm.invoke(messages)
        logger.info("AI summary generated for profile '%s'", active_profile)
        return extract_text_content(response.content)
    except Exception as e:
        logger.error("AI summary generation failed for '%s': %s", active_profile, e)
        return f'Unable to generate AI summary: {str(e)}'

def generate_ai_summary(user_data, active_profile, metrics, provider=None, model_name=None, api_key=None):
    from src.storage import load_user_api_keys
    demographics = user_data.get('demographics', {})
    metrics_json = json.dumps(metrics, sort_keys=True)
    active_provider = provider or st.session_state.get('selected_provider', get_default_provider())
    active_model = model_name or st.session_state.get('selected_model', '')
    active_key = api_key or load_user_api_keys().get(active_provider)
    return fetch_cached_ai_summary(
        active_profile,
        metrics_json,
        demographics.get('fitness_goal', 'N/A'),
        demographics.get('fitness_level', 'N/A'),
        active_provider,
        active_model,
        api_key=active_key,
    )

def build_trainer_system_prompt(demographics, active_profile):
    add_context = demographics.get('additional_context')
    add_context_note = f'\n- Additional Context / Notes: {add_context}' if add_context else ''
    tmpl = PROMPTS.get('trainer_system_prompt', '')
    return tmpl.format(
        active_profile=active_profile,
        gender=demographics.get('gender', 'Not Specified'),
        age=demographics.get('age', 'N/A'),
        height=demographics.get('height', 'N/A'),
        weight=demographics.get('weight', 'N/A'),
        nationality=demographics.get('nationality', 'N/A'),
        fitness_goal=demographics.get('fitness_goal', 'N/A'),
        fitness_level=demographics.get('fitness_level', 'Beginner'),
        equipment=demographics.get('equipment', 'N/A'),
        days_per_week=demographics.get('days_per_week', 4),
        rest_days=', '.join(demographics.get('rest_days', [])) if demographics.get('rest_days') else 'None',
        dietary_preference=demographics.get('dietary_preference', 'Standard'),
        session_duration=demographics.get('session_duration', '45 minutes'),
        injuries=demographics.get('injuries', 'None'),
        additional_context_note=add_context_note,
    ).strip()

MAX_CHAT_CONTEXT = 20

def get_chat_messages(chat_history, demographics, active_profile):
    sys_prompt = build_trainer_system_prompt(demographics, active_profile)
    lc_messages = [SystemMessage(content=sys_prompt)]
    recent_history = chat_history[-MAX_CHAT_CONTEXT:]
    if len(chat_history) > MAX_CHAT_CONTEXT:
        logger.info('Chat truncated: %d -> %d messages for LLM context', len(chat_history), MAX_CHAT_CONTEXT)
    for m in recent_history:
        text_content = extract_text_content(m.get('content', ''))
        if m.get('role') == 'user':
            lc_messages.append(HumanMessage(content=text_content))
        else:
            lc_messages.append(AIMessage(content=text_content))
    return lc_messages

def stream_chat_response(lc_messages, provider, model_name, api_key=None, temperature=0.7):
    logger.info('Streaming chat via %s (%s)', provider, model_name)
    llm = get_llm_instance(provider, model_name, api_key=api_key, temperature=temperature, num_ctx=2048)
    for chunk in llm.stream(lc_messages):
        text = extract_text_content(chunk.content)
        if text:
            yield text
