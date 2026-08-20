import base64
import datetime
import json
import logging
import os
import re
import time
import matplotlib.pyplot as plt
import streamlit as st
import yaml
from dotenv import load_dotenv
from src.auth import authenticate_user, render_logout
from src.ai import (
    SUPPORTED_PROVIDERS, PROVIDER_ENV_KEYS,
    get_available_providers, get_default_provider,
    validate_and_fetch_models, fetch_provider_models,
    generate_approved_schema, generate_ai_summary,
    get_chat_messages, stream_chat_response, format_error_message,
    extract_text_content,
)
from src.storage import (
    list_profiles, get_profile_filepath,
    load_profile_data, save_profile_data, rename_profile,
    load_user_api_keys, save_user_api_keys,
)

from src.metrics import (
    WEEKDAYS, parse_numeric_value, validate_demographics_ranges,
    calculate_profile_metrics, extract_timeline_series,
)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

NATIONALITIES = [
    'Indian',
    'Afghan', 'Albanian', 'Algerian', 'American', 'Andorran', 'Angolan', 'Argentine', 'Armenian', 'Australian',
    'Austrian', 'Azerbaijani', 'Bahamian', 'Bahraini', 'Bangladeshi', 'Barbadian', 'Belarusian', 'Belgian',
    'Belizean', 'Beninese', 'Bhutanese', 'Bolivian', 'Bosnian', 'Botswanan', 'Brazilian', 'British', 'Bruneian',
    'Bulgarian', 'Burkinabe', 'Burmese', 'Burundian', 'Cambodian', 'Cameroonian', 'Canadian', 'Cape Verdean',
    'Central African', 'Chadian', 'Chilean', 'Chinese', 'Colombian', 'Comoran', 'Congolese', 'Costa Rican',
    'Croatian', 'Cuban', 'Cypriot', 'Czech', 'Danish', 'Djiboutian', 'Dominican', 'Dutch', 'East Timorese',
    'Ecuadorian', 'Egyptian', 'Emirati', 'Equatorial Guinean', 'Eritrean', 'Estonian', 'Ethiopian', 'Fijian',
    'Finnish', 'French', 'Gabonese', 'Gambian', 'Georgian', 'German', 'Ghanaian', 'Greek', 'Grenadian',
    'Guatemalan', 'Guinean', 'Guyanese', 'Haitian', 'Honduran', 'Hungarian', 'Icelandic', 'Indonesian', 'Iranian',
    'Iraqi', 'Irish', 'Israeli', 'Italian', 'Ivorian', 'Jamaican', 'Japanese', 'Jordanian', 'Kazakhstani',
    'Kenyan', 'Kuwaiti', 'Kyrgyz', 'Laotian', 'Latvian', 'Lebanese', 'Liberian', 'Libyan', 'Liechtensteiner',
    'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malawian', 'Malaysian', 'Maldivian', 'Malian',
    'Maltese', 'Mauritanian', 'Mauritian', 'Mexican', 'Micronesian', 'Moldovan', 'Monegasque', 'Mongolian',
    'Montenegrin', 'Moroccan', 'Mozambican', 'Namibian', 'Nauruan', 'Nepalese', 'New Zealander', 'Nicaraguan',
    'Nigerian', 'Nigerien', 'North Korean', 'Norwegian', 'Omani', 'Pakistani', 'Palauan', 'Palestinian',
    'Panamanian', 'Papua New Guinean', 'Paraguayan', 'Peruvian', 'Filipino', 'Polish', 'Portuguese', 'Qatari',
    'Romanian', 'Russian', 'Rwandan', 'Saint Lucian', 'Salvadoran', 'Samoan', 'San Marinese', 'Saudi', 'Senegalese',
    'Serbian', 'Seychellois', 'Sierra Leonean', 'Singaporean', 'Slovak', 'Slovenian', 'Solomon Islander', 'Somali',
    'South African', 'South Korean', 'South Sudanese', 'Spanish', 'Sri Lankan', 'Sudanese', 'Surinamese', 'Swazi',
    'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Tajik', 'Tanzanian', 'Thai', 'Togolese', 'Tongan', 'Trinidadian',
    'Tunisian', 'Turkish', 'Turkmen', 'Tuvaluan', 'Ugandan', 'Ukrainian', 'Uruguayan', 'Uzbek', 'Vanuatu',
    'Venezuelan', 'Vietnamese', 'Yemeni', 'Zambian', 'Zimbabwean'
]
AUTH_CONFIG_PATH = os.path.join(_PROJECT_ROOT, 'config', 'auth.yaml')
logger = logging.getLogger('gym_assister.app')
MAX_LLM_CALLS = 20
RATE_WINDOW_SECONDS = 600

def check_rate_limit():
    now = time.time()
    key = '_llm_call_timestamps'
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key] = [t for t in st.session_state[key] if now - t < RATE_WINDOW_SECONDS]
    if len(st.session_state[key]) >= MAX_LLM_CALLS:
        logger.warning('Rate limit exceeded: %d calls in %d seconds', MAX_LLM_CALLS, RATE_WINDOW_SECONDS)
        raise RuntimeError(f'Rate limit exceeded: {MAX_LLM_CALLS} LLM calls per {RATE_WINDOW_SECONDS // 60} minutes. Please wait before sending more requests.')
    st.session_state[key].append(now)


st.set_page_config(page_title='Gym Assister', page_icon='🏋️\u200d♂️', layout='wide')
st.markdown('''
    <style>
    div[data-testid="stPopover"] button svg {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] {
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stPopover"] > button {
        min-width: 0 !important;
        width: 100% !important;
        padding: 2px 0px !important;
        font-size: 15px !important;
        margin: 0 !important;
        height: 38px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] div[data-testid="column"] {
        min-width: 0 !important;
    }
    </style>
    ''', unsafe_allow_html=True)


@st.dialog('LLM Provider Options')
def options_dialog():
    provider_names = get_available_providers()
    default_prov = get_default_provider()
    active_provider = st.session_state.get('selected_provider', default_prov)
    if active_provider not in provider_names:
        active_provider = default_prov
    active_model = st.session_state.get('selected_model')
    auth_username = st.session_state.get('auth_username')
    user_keys = load_user_api_keys(auth_username)

    p_idx = provider_names.index(active_provider) if active_provider in provider_names else 0
    sel_provider = st.selectbox('LLM Provider', provider_names, index=p_idx, key='dialog_opt_provider_sel')
    current_key = user_keys.get(sel_provider, '')
    fetched_key = f'fetched_models_{sel_provider}_{auth_username}'

    is_valid = False
    p_models = []
    new_key = st.text_input(f'{sel_provider} API Key', value=current_key, type='password', key=f'dialog_opt_api_key_{auth_username}_{sel_provider}').strip()
    if new_key:
        if fetched_key in st.session_state and st.session_state.get(f'_validated_key_{sel_provider}_{auth_username}') == new_key:
            p_models = st.session_state[fetched_key]
            is_valid = bool(p_models)
        else:
            valid, models, _ = validate_and_fetch_models(sel_provider, api_key=new_key)
            if valid and models:
                is_valid = True
                p_models = models
                st.session_state[fetched_key] = models
                st.session_state[f'_validated_key_{sel_provider}_{auth_username}'] = new_key
            else:
                is_valid = False
                p_models = []
                st.session_state[fetched_key] = []
                st.session_state[f'_validated_key_{sel_provider}_{auth_username}'] = None
    else:
        st.session_state[fetched_key] = []
        st.session_state[f'_validated_key_{sel_provider}_{auth_username}'] = None

    sel_model = None
    if is_valid and p_models:
        m_idx = p_models.index(active_model) if active_model in p_models else 0
        sel_model = st.selectbox('Select Model', p_models, index=m_idx, key='dialog_opt_model_sel')
    else:
        st.selectbox('Select Model', ['Enter valid API key to load models'], disabled=True, key='dialog_opt_model_disabled')

    st.divider()
    done_disabled = not (is_valid and p_models and sel_model)
    if st.button('Done', disabled=done_disabled, type='primary', use_container_width=True, key=f'dialog_opt_done_btn_{auth_username}'):
        user_keys[sel_provider] = new_key
        save_user_api_keys(user_keys, auth_username)
        st.session_state['selected_provider'] = sel_provider
        st.session_state['selected_model'] = sel_model
        st.rerun()



def render_top_header(user_data=None, active_profile=None):
    _logo_path = os.path.join(_PROJECT_ROOT, 'assets', 'logo.jpg')
    if not os.path.exists(_logo_path):
        _logo_path = os.path.join(_PROJECT_ROOT, 'data', 'logo.jpg')
    _img_tag = ''
    if os.path.exists(_logo_path):
        with open(_logo_path, 'rb') as _f:
            _img_b64 = base64.b64encode(_f.read()).decode()
        _img_tag = (
            f'<img src="data:image/jpeg;base64,{_img_b64}" '
            'style="width:420px;max-width:100%;height:auto;object-fit:cover;'
            'border-radius:12px;border:1px solid rgba(139,92,246,0.35);box-shadow:0 8px 24px rgba(0,0,0,0.35);margin:0 auto;display:block;" />'
        )

    header_html = f'''
        <div style="text-align:center; padding:0.1rem 0 0.5rem 0;">
            <div style="font-size:2.2rem; font-weight:800; letter-spacing:-0.02em; margin-bottom:0.75rem;
                background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 100%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Gym Assister
            </div>
            {_img_tag}
        </div>
    '''
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown('---')



@st.dialog('Customize your Fitness Profile')
def onboarding_dialog():
    st.write('Welcome to Gym Assister! Create a profile and setup your AI trainer.')
    profile_name_input = st.text_input('Profile Name*', value='', placeholder='The Great Khali', key='ob_name')
    is_name_empty = not profile_name_input.strip()
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Non-Binary', 'Prefer not to say'], key='ob_gender')
        age = st.number_input('Age', min_value=12, max_value=100, value=25, step=1, key='ob_age')
        height = st.text_input('Height (e.g. 175 cm or 5\'9")', value='175 cm', key='ob_height')
        weight = st.text_input('Weight (e.g. 70 kg or 154 lbs)', value='70 kg', key='ob_weight')
        nationality = st.selectbox('Nationality', NATIONALITIES, index=0, key='ob_nationality')
        fitness_level = st.selectbox('Fitness Experience Level', ['Beginner', 'Intermediate', 'Advanced'], key='ob_level')
    with col2:
        fitness_goal = st.selectbox('Fitness Goal', ['Bulk (Muscle Gain)', 'Cut (Fat Loss)', 'Recomposition', 'Strength & Power', 'Endurance & Fitness', 'Flexibility & Mobility'], key='ob_goal')
        equipment = st.selectbox('Equipment Availability', ['Full Gym Access', 'Dumbbells Only', 'Bodyweight / Home', 'Barbell & Plates', 'Resistance Bands'], key='ob_equipment')
        days_per_week = st.slider('Days Available Per Week', min_value=1, max_value=7, value=4, key='ob_days')
        dietary_preference = st.selectbox('Dietary Preference', ['Standard', 'Vegetarian', 'Vegan', 'Eggetarian', 'Keto', 'High Protein', 'Halal'], key='ob_diet')
        duration_mins = st.number_input('Preferred Session Duration (minutes)', min_value=10, max_value=120, value=45, step=5, key='ob_duration_mins')
        session_duration = f'{duration_mins} minutes'
    rest_days = []
    if days_per_week < 7:
        required_rest_days = 7 - days_per_week
        default_rest = WEEKDAYS[-required_rest_days:]
        rest_days = st.multiselect('Select Rest Days', WEEKDAYS, default=default_rest, max_selections=required_rest_days, key=f'ob_rest_days_{days_per_week}')
    injuries = st.text_input('Medical Conditions / Injuries / Limitations', value='None', key='ob_injuries')
    additional_context = st.text_area('Additional Context / Special Instructions', value='', placeholder='e.g. Focus on shoulder hypertrophy, preference for high protein Indian recipes, short rest times...', key='ob_additional_context')
    st.divider()
    submitted = st.button('Save Profile & Get Started', type='primary', use_container_width=True, disabled=is_name_empty, key='ob_submit')
    if is_name_empty:
        st.caption('Please enter a Profile Name above to enable saving.')
    if submitted:
        required_rest_days = 7 - days_per_week
        range_errors = validate_demographics_ranges(height, weight)
        if not profile_name_input.strip() or not nationality.strip() or (not height.strip()) or (not weight.strip()):
            st.error('Please fill in all required profile fields.')
        elif range_errors:
            for err in range_errors:
                st.error(err)
        elif days_per_week < 7 and len(rest_days) != required_rest_days:
            st.error(f'Please select exactly {required_rest_days} rest day(s).')
        else:
            filepath, safe_name = get_profile_filepath(profile_name_input)
            profile_data = load_profile_data(safe_name)
            existing_demo = profile_data.get('demographics')
            overwrite_key = f'ob_overwrite_confirmed_{safe_name}'
            if existing_demo and not st.session_state.get(overwrite_key, False):
                st.warning(
                    f"Profile **'{safe_name}'** already has saved demographics. "
                    "Click **Save Profile & Get Started** again to overwrite, or choose a different name."
                )
                st.session_state[overwrite_key] = True
                return
            st.session_state.pop(overwrite_key, None)
            profile_data['profile_name'] = safe_name
            profile_data['demographics'] = {
                'gender': gender,
                'age': int(age),
                'height': height.strip(),
                'weight': weight.strip(),
                'nationality': nationality.strip(),
                'fitness_goal': fitness_goal,
                'fitness_level': fitness_level,
                'equipment': equipment,
                'days_per_week': int(days_per_week),
                'rest_days': rest_days if days_per_week < 7 else [],
                'dietary_preference': dietary_preference,
                'session_duration': session_duration,
                'injuries': injuries.strip(),
                'additional_context': additional_context.strip(),
                'created_at': existing_demo.get('created_at', datetime.date.today().isoformat()) if existing_demo else datetime.date.today().isoformat(),
            }
            save_profile_data(safe_name, profile_data)
            st.session_state['active_profile'] = safe_name
            st.session_state['show_add_profile'] = False
            st.success(f"Profile '{safe_name}' saved!")
            st.rerun()


def render_sidebar(user_data, active_profile):
    profiles = list_profiles()
    has_approved_plan = bool(user_data.get('approved_schema'))
    with st.sidebar:
        if profiles:
            current_index = profiles.index(active_profile) if active_profile in profiles else 0
            st.caption('Active Profile')
            col_sel, col_rename, col_del = st.columns([3.6, 1.2, 1.2], gap='small')
            with col_sel:
                selected_profile = st.selectbox('Switch Active Profile', profiles, index=current_index, key='sidebar_profile_selector', label_visibility='collapsed')
                if selected_profile != active_profile:
                    st.session_state['active_profile'] = selected_profile
                    st.session_state['show_summary_dashboard'] = False
                    st.rerun()
            with col_rename:
                with st.popover('✏️', help='Rename active profile', use_container_width=True):
                    st.markdown('#### Rename Profile')
                    new_pname = st.text_input('New Name', value=active_profile, key='rename_input_field')
                    if st.button('Confirm', key='rename_confirm_btn', type='primary', use_container_width=True):
                        success, res = rename_profile(active_profile, new_pname)
                        if success:
                            st.session_state['active_profile'] = res
                            st.success(f"Renamed to '{res}'!")
                            st.rerun()
                        else:
                            st.error(res)
            with col_del:
                with st.popover('🗑️', help='Delete active profile', use_container_width=True):
                    st.markdown('**Are you sure?**')
                    st.write(f"This will permanently delete profile **'{active_profile}'**.")
                    if st.button('Confirm Delete', key='del_confirm_btn', type='primary', use_container_width=True):
                        filepath, safe_name = get_profile_filepath(active_profile)
                        if filepath and os.path.exists(filepath):
                            os.remove(filepath)
                        st.session_state['active_profile'] = None
                        st.session_state['show_summary_dashboard'] = False
                        st.rerun()
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            if st.button('➕ Add Profile', type='primary', use_container_width=True):
                st.session_state['show_add_profile'] = True
                st.rerun()
        with col_sb2:
            if st.button('Summary', type='secondary', use_container_width=True, disabled=not has_approved_plan, help='Approve plan first to unlock summary'):
                st.session_state['show_summary_dashboard'] = True
                st.rerun()
        st.divider()
        demographics = user_data.get('demographics', {})
        st.subheader(f"👤 {active_profile}'s Details")
        if demographics:
            st.write(f"**Gender:** {demographics.get('gender', 'N/A')}")
            st.write(f"**Age:** {demographics.get('age')}")
            st.write(f"**Height:** {demographics.get('height')}")
            st.write(f"**Weight:** {demographics.get('weight')}")
            st.write(f"**Nationality:** {demographics.get('nationality')}")
            st.write(f"**Goal:** {demographics.get('fitness_goal')}")
            st.write(f"**Level:** {demographics.get('fitness_level', 'N/A')}")
            st.write(f"**Equipment:** {demographics.get('equipment')}")
            st.write(f"**Workout Days:** {demographics.get('days_per_week', 'N/A')} days/week")
            rest_days_list = demographics.get('rest_days', [])
            if rest_days_list:
                st.write(f"**Rest Days:** {', '.join(rest_days_list)}")
            st.write(f"**Diet:** {demographics.get('dietary_preference', 'N/A')}")
            st.write(f"**Session:** {demographics.get('session_duration', 'N/A')}")
            if demographics.get('injuries') and demographics.get('injuries') != 'None':
                st.write(f"**Injuries:** {demographics.get('injuries')}")
            if demographics.get('additional_context'):
                st.write(f"**Notes:** {demographics.get('additional_context')}")
        if user_data.get('approved_schema'):
            if st.button('Modify Plan', type='secondary', use_container_width=True):
                user_data['approved_schema'] = None
                user_data['weekly_logs'] = {}
                save_profile_data(active_profile, user_data)
                st.session_state['show_summary_dashboard'] = False
                st.rerun()

        st.divider()
        st.subheader('LLM Settings')
        provider_names = get_available_providers()
        default_prov = get_default_provider()
        active_provider = st.session_state.get('selected_provider', default_prov)
        if active_provider not in provider_names:
            active_provider = default_prov
            st.session_state['selected_provider'] = active_provider
        active_model = st.session_state.get('selected_model')
        model_display = f'{active_model}' if active_model else 'Not Configured'
        st.caption(f'Active: **{active_provider}** ({model_display})')
        if st.button('Options', key='sidebar_open_options_dialog', use_container_width=True, type='primary'):
            options_dialog()
        chat_history = user_data.get('chat_history', [])
        if st.button('Clear History', key=f'sidebar_clear_chat_btn_{active_profile}', type='secondary', use_container_width=True, disabled=not bool(chat_history)):
            user_data['chat_history'] = []
            save_profile_data(active_profile, user_data)
            st.toast('Chat history cleared!', icon='🧹')
            st.rerun()




def render_summary_dashboard(user_data, active_profile):
    metrics = calculate_profile_metrics(user_data)
    demographics = user_data.get('demographics', {})
    col_title, col_back = st.columns([3, 1])
    with col_title:
        st.subheader(f'Summary')
    with col_back:
        if st.button('Back to Schedule', use_container_width=True):
            st.session_state['show_summary_dashboard'] = False
            st.rerun()
    st.markdown('---')
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(label='Workout Adherence', value=f"{metrics['adherence_pct']}%", delta=f"{metrics['completed_exercises']}/{metrics['total_assigned']} Done" if metrics['total_assigned'] > 0 else 'No logs yet')
    with mcol2:
        cal_delta = metrics['avg_calories'] - metrics['target_calories'] if metrics['target_calories'] > 0 else 0
        st.metric(label='Avg Daily Calories', value=f"{metrics['avg_calories']} kcal", delta=f'{cal_delta:+d} kcal vs Target' if metrics['target_calories'] > 0 else 'Target 0')
    with mcol3:
        st.metric(label='Weight Trajectory', value=f"{metrics['latest_weight']} kg", delta=f"{metrics['weight_change']:+.1f} kg overall")
    with mcol4:
        st.metric(label='Schedule & Rest Days', value=f"{metrics['total_rest_count']} Total Rest Days", delta=f"{metrics['scheduled_rest_count']} Scheduled + {metrics['additional_rest_count']} Unlogged")
    st.caption(f"Plan Timeline — Created: **{metrics['first_date']}** | Latest Data: **{metrics['latest_date']}**")
    st.markdown('---')

    cal_points, wt_points = extract_timeline_series(user_data)
    target_cals = metrics.get('target_calories', 0)
    initial_wt = metrics.get('initial_weight', 0.0)

    st.markdown('### 📊 Progress Analytics')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('##### Calories Consumed vs Days')
        if cal_points:
            fig, ax = plt.subplots(figsize=(6, 3.6), facecolor='#0e1117')
            ax.set_facecolor('#161b22')
            x_labels = [p['date'] for p in cal_points]
            x_idx = list(range(len(x_labels)))
            y_cals = [p['value'] for p in cal_points]
            ax.plot(x_idx, y_cals, marker='o', color='#38bdf8', linewidth=2.2, markersize=5.5, label='Logged Calories')
            if target_cals > 0:
                ax.axhline(target_cals, color='#f43f5e', linestyle='--', linewidth=1.5, alpha=0.85, label=f'Target ({target_cals} kcal)')
            ax.set_ylabel('Calories (kcal)', color='#94a3b8', fontsize=9.5)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(x_labels)
            ax.tick_params(axis='x', colors='#cbd5e1', labelsize=8, rotation=35)
            ax.tick_params(axis='y', colors='#cbd5e1', labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8, loc='best')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info('No daily calorie logs recorded yet.')
    with c2:
        st.markdown('##### Weight vs Days')
        if wt_points:
            fig, ax = plt.subplots(figsize=(6, 3.6), facecolor='#0e1117')
            ax.set_facecolor('#161b22')
            x_labels = [p['date'] for p in wt_points]
            x_idx = list(range(len(x_labels)))
            y_wts = [p['value'] for p in wt_points]
            ax.plot(x_idx, y_wts, marker='s', color='#a855f7', linewidth=2.2, markersize=5.5, label='Logged Weight')
            if initial_wt > 0:
                ax.axhline(initial_wt, color='#eab308', linestyle='--', linewidth=1.5, alpha=0.85, label=f'Initial ({initial_wt:.1f} kg)')
            ax.set_ylabel('Weight (kg)', color='#94a3b8', fontsize=9.5)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(x_labels)
            ax.tick_params(axis='x', colors='#cbd5e1', labelsize=8, rotation=35)
            ax.tick_params(axis='y', colors='#cbd5e1', labelsize=8)
            ax.grid(True, linestyle=':', alpha=0.25, color='#475569')
            for spine in ax.spines.values():
                spine.set_color('#334155')
            ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8, loc='best')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info('No daily weight logs recorded yet.')
    st.markdown('---')

    st.markdown('### 🤖 AI Trainer Progress Report')
    with st.spinner(f'Generating progress analysis for {active_profile}...'):
        try:
            check_rate_limit()
        except RuntimeError as e:
            st.warning(str(e))
            return
        active_provider = st.session_state.get('selected_provider', get_default_provider())
        active_model = st.session_state.get('selected_model')
        auth_username = st.session_state.get('auth_username')
        active_api_key = load_user_api_keys(auth_username).get(active_provider)
        report = generate_ai_summary(user_data, active_profile, metrics, provider=active_provider, model_name=active_model, api_key=active_api_key)
        with st.container(border=True):
            st.markdown(report)


def render_chat_interface(user_data, active_profile):
    active_provider = st.session_state.get('selected_provider', get_default_provider())
    active_model = st.session_state.get('selected_model')
    auth_username = st.session_state.get('auth_username')
    active_api_key = load_user_api_keys(auth_username).get(active_provider)
    demographics = user_data['demographics']
    chat_history = user_data.get('chat_history', [])

    is_llm_ready = bool(active_api_key and active_model)
    st.subheader('Personal Trainer')

    if st.session_state.get('is_approving', False):

        st.info('Generating your personalized workout plan...')
        with st.spinner(f'Analyzing plan via {active_provider} ({active_model})...'):
            try:
                check_rate_limit()
                schema = generate_approved_schema(user_data['chat_history'], active_provider, active_model, api_key=active_api_key)
                user_data['approved_schema'] = schema
                user_data['weekly_logs'] = {}
                save_profile_data(active_profile, user_data)
                st.session_state['is_approving'] = False
                st.success('Plan approved and schema generated!')
                st.rerun()
            except Exception as e:
                st.session_state['is_approving'] = False
                st.toast(format_error_message(e, active_provider, active_model), icon='⚠️')

    for msg in chat_history:
        with st.chat_message(msg['role']):
            st.write(extract_text_content(msg.get('content', '')))

    is_busy = st.session_state.get('chat_in_progress', False) or st.session_state.get('is_approving', False)
    if not is_llm_ready:
        st.chat_input('Chat is disabled. Please configure a valid API key in the left sidebar...', disabled=True)
        return
    if is_busy:
        st.chat_input('Trainer is generating a response, please wait...', disabled=True)
        return

    if (prompt := st.chat_input('Ask advice, or type /plan, /approve, /summarize...')):
        st.session_state['chat_in_progress'] = True
        try:
            cmd = prompt.strip().lower()
            if cmd == '/summarize':
                if user_data.get('approved_schema'):
                    st.session_state['show_summary_dashboard'] = True
                    st.rerun()
                else:
                    st.warning('Summary Dashboard is locked until your workout & diet plan is approved!')
                    return
            if cmd == '/approve':
                if not user_data.get('chat_history'):
                    st.warning('Please chat with your trainer or type /plan to discuss your plan first before approving.')
                    return
                st.session_state['is_approving'] = True
                st.rerun()
            if cmd == '/plan':
                display_prompt = '/plan'
                actual_prompt = 'Please generate a complete, structured, and personalized workout and diet plan based on my profile demographics, fitness level, and goals.'
            else:
                display_prompt = prompt
                actual_prompt = prompt
            user_data['chat_history'].append({'role': 'user', 'content': display_prompt})
            save_profile_data(active_profile, user_data)
            with st.chat_message('user'):
                st.write(display_prompt)
            prompt_for_ai = [{'role': m['role'], 'content': actual_prompt if (m['role'] == 'user' and m['content'] == '/plan' and i == len(user_data['chat_history']) - 1) else m['content']} for i, m in enumerate(user_data['chat_history'])]
            lc_messages = get_chat_messages(prompt_for_ai, demographics, active_profile)
            with st.chat_message('assistant'):
                try:
                    check_rate_limit()
                    response_text = st.write_stream(stream_chat_response(lc_messages, active_provider, active_model, api_key=active_api_key))
                    clean_text = extract_text_content(response_text)
                    user_data['chat_history'].append({'role': 'assistant', 'content': clean_text})
                    save_profile_data(active_profile, user_data)
                except Exception as e:
                    st.toast(format_error_message(e, active_provider, active_model), icon='⚠️')
        finally:
            st.session_state['chat_in_progress'] = False



def render_dashboard(user_data, active_profile):
    schema = user_data['approved_schema']
    demographics = user_data.get('demographics', {})
    rest_days = demographics.get('rest_days', [])
    st.subheader(f'Dashboard')
    created_at_str = demographics.get('created_at', datetime.date.today().isoformat())
    try:
        plan_start_date = datetime.date.fromisoformat(created_at_str)
    except Exception:
        plan_start_date = datetime.date.today()
    default_sel_date = max(datetime.date.today(), plan_start_date)
    selected_date = st.date_input('Select Week Target Date', value=default_sel_date, min_value=plan_start_date)
    monday_start = selected_date - datetime.timedelta(days=selected_date.weekday())
    sunday_end = monday_start + datetime.timedelta(days=6)
    week_label = f"{monday_start.strftime('%b %d, %Y')} – {sunday_end.strftime('%b %d, %Y')}"
    week_key = f'week_{monday_start.isoformat()}'
    weekly_logs = user_data.get('weekly_logs', {})
    week_log = weekly_logs.get(week_key, {'exercises': {}, 'daily_metrics': {}})
    st.markdown(f'### Week of **{week_label}**')
    with st.form('weekly_tracking_form'):
        diet_info = schema.get('diet', {})
        target_cals = diet_info.get('calories', 0)
        target_protein = diet_info.get('protein', 0)
        st.markdown(f'#### 🎯 Weekly Targets\n\n**Daily Calories:** {target_cals} kcal &nbsp;&nbsp;|&nbsp;&nbsp; **Daily Protein:** {target_protein} g')
        workout_days = schema.get('workout_days', [])
        active_days = [d for d in WEEKDAYS if d not in rest_days] or WEEKDAYS
        tabs = st.tabs(active_days)
        updated_exercises = dict(week_log.get('exercises', {}))
        updated_metrics = dict(week_log.get('daily_metrics', {}))
        day_workout_map = {}
        for idx, w_day in enumerate(active_days):
            if idx < len(workout_days):
                day_workout_map[w_day] = workout_days[idx]
        for day_idx, weekday_name in enumerate(active_days):
            with tabs[day_idx]:
                col_day_workout, col_day_metrics = st.columns([3, 2])
                with col_day_workout:
                    workout_info = day_workout_map.get(weekday_name, None)
                    if workout_info:
                        day_title = workout_info.get('day', weekday_name)
                        st.markdown(f'#### 💪 {day_title}')
                        exercises = workout_info.get('exercises', [])
                        for ex in exercises:
                            ex_name = ex.get('name', 'Exercise')
                            ex_sets = ex.get('sets', 0)
                            ex_reps = ex.get('reps', 0)
                            ex_id = f'{weekday_name}_{ex_name}'
                            is_checked = updated_exercises.get(ex_id, False)
                            cb_val = st.checkbox(f'**{ex_name}** — {ex_sets} sets × {ex_reps} reps', value=is_checked, key=f'{active_profile}_{week_key}_{ex_id}')
                            updated_exercises[ex_id] = cb_val
                    else:
                        st.caption(f'No workout assigned for {weekday_name}.')
                with col_day_metrics:
                    st.markdown(f'#### Metrics')
                    day_metric = updated_metrics.get(weekday_name, {'calories': 0, 'weight': 0.0})
                    raw_cals = int(day_metric.get('calories', 0) or 0)
                    default_cals = min(max(raw_cals, 500), 7000) if raw_cals > 0 else max(500, min(target_cals, 7000))
                    raw_wt = float(day_metric.get('weight', 0.0) or 0.0)
                    profile_wt = parse_numeric_value(demographics.get('weight', '70')) or 70.0
                    profile_wt_clamped = min(max(profile_wt, 30.0), 200.0)
                    default_wt = min(max(raw_wt, 30.0), 200.0) if raw_wt > 0.0 else profile_wt_clamped
                    cals_in = st.number_input(f'Calories Consumed', min_value=500, max_value=7000, value=default_cals, step=50, key=f'{active_profile}_{week_key}_{weekday_name}_cals')
                    weight_in = st.number_input(f'Logged Weight', min_value=30.0, max_value=200.0, value=default_wt, step=0.5, key=f'{active_profile}_{week_key}_{weekday_name}_weight')
                    updated_metrics[weekday_name] = {'calories': cals_in, 'weight': weight_in}
        st.divider()
        save_submitted = st.form_submit_button('💾 Save Tracking Data', type='primary', use_container_width=True)
        if save_submitted:
            week_log['exercises'] = updated_exercises
            week_log['daily_metrics'] = updated_metrics
            weekly_logs[week_key] = week_log
            user_data['weekly_logs'] = weekly_logs
            save_profile_data(active_profile, user_data)
            st.success(f'Tracking data for week of {week_label} saved.')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    authenticator = authenticate_user()
    auth_username = st.session_state.get('auth_username', 'default')
    profiles = list_profiles(auth_username)
    with st.sidebar:

        render_logout(authenticator)
    if not profiles or st.session_state.get('show_add_profile', False):
        render_top_header(None, None)
        if st.button('Please create a profile to continue.', type='tertiary', key='prompt_create_profile_link'):
            onboarding_dialog()
        if not st.session_state.get('_onboarding_shown', False):
            st.session_state['_onboarding_shown'] = True
            onboarding_dialog()
        return
    active_profile = st.session_state.get('active_profile')
    if active_profile not in profiles:
        active_profile = profiles[0]
        st.session_state['active_profile'] = active_profile
    user_data = load_profile_data(active_profile)
    render_top_header(user_data, active_profile)
    if not user_data.get('demographics'):
        if st.button(f"Please complete setup for profile '{active_profile}' to continue.", type='tertiary', key='prompt_setup_profile_link'):
            onboarding_dialog()
        if not st.session_state.get(f'_onboarding_shown_{active_profile}', False):
            st.session_state[f'_onboarding_shown_{active_profile}'] = True
            onboarding_dialog()
    else:
        render_sidebar(user_data, active_profile)
        if st.session_state.get('show_summary_dashboard', False):
            render_summary_dashboard(user_data, active_profile)
        elif user_data.get('approved_schema'):
            render_dashboard(user_data, active_profile)
        else:
            render_chat_interface(user_data, active_profile)


if __name__ == '__main__':
    main()
