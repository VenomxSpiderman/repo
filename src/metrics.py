import datetime
import re

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def parse_numeric_value(val_str):
    try:
        matches = re.findall('[-+]?\\d*\\.\\d+|\\d+', str(val_str))
        if matches:
            return float(matches[0])
    except Exception:
        pass
    return None

def validate_demographics_ranges(height_str, weight_str):
    h_val = parse_numeric_value(height_str)
    w_val = parse_numeric_value(weight_str)
    errors = []
    if h_val is None or not 100.0 <= h_val <= 230.0:
        errors.append('Height must be a valid number between 100 cm and 230 cm (or ~3\'3" to 7\'6").')
    if w_val is None or not 30.0 <= w_val <= 200.0:
        errors.append('Weight must be a valid number between 30 kg and 200 kg (or ~66 lbs to 440 lbs).')
    return errors

def calculate_profile_metrics(user_data, today_override=None):
    demographics = user_data.get('demographics', {})
    schema = user_data.get('approved_schema', {})
    weekly_logs = user_data.get('weekly_logs', {})
    created_at_str = demographics.get('created_at', datetime.date.today().isoformat())
    try:
        first_date = datetime.date.fromisoformat(created_at_str)
    except Exception:
        first_date = datetime.date.today()
    total_assigned = 0
    total_completed = 0
    logged_calories = []
    logged_weights = []
    additional_rest_days = 0
    target_cals = schema.get('diet', {}).get('calories', 0) if schema else 0
    scheduled_rest_days = demographics.get('rest_days', [])
    logged_dates = []
    today = today_override or datetime.date.today()
    current_monday = today - datetime.timedelta(days=today.weekday())
    current_week_key = f'week_{current_monday.isoformat()}'
    for week_key, week_data in weekly_logs.items():
        if week_key.startswith('week_'):
            try:
                w_date = datetime.date.fromisoformat(week_key.replace('week_', ''))
                logged_dates.append(w_date)
            except Exception:
                pass
        exercises = week_data.get('exercises', {})
        daily_metrics = week_data.get('daily_metrics', {})
        for ex_id, is_done in exercises.items():
            total_assigned += 1
            if is_done:
                total_completed += 1
        if week_key == current_week_key:
            today_idx = today.weekday()
            for day_idx, day_name in enumerate(WEEKDAYS):
                if day_idx > today_idx:
                    continue
                if day_name not in scheduled_rest_days:
                    has_exercise_checked = any((is_done for ex_id, is_done in exercises.items() if ex_id.startswith(f'{day_name}_')))
                    day_metrics = daily_metrics.get(day_name, {})
                    has_metrics_logged = day_metrics.get('calories', 0) > 0 or day_metrics.get('weight', 0.0) > 0.0
                    if not has_exercise_checked and (not has_metrics_logged):
                        additional_rest_days += 1
        for day_name in WEEKDAYS:
            metrics = daily_metrics.get(day_name, {})
            cals = metrics.get('calories', 0)
            wt = metrics.get('weight', 0.0)
            if cals > 0:
                logged_calories.append(cals)
            if wt > 0:
                logged_weights.append(wt)
    latest_date = max(logged_dates) + datetime.timedelta(days=6) if logged_dates else first_date
    adherence_pct = round(total_completed / total_assigned * 100, 1) if total_assigned > 0 else 0.0
    avg_cals = round(sum(logged_calories) / len(logged_calories)) if logged_calories else 0
    initial_wt_str = str(demographics.get('weight', '0'))
    try:
        initial_wt = float(re.findall('[-+]?\\d*\\.\\d+|\\d+', initial_wt_str)[0])
    except Exception:
        initial_wt = 0.0
    latest_wt = logged_weights[-1] if logged_weights else initial_wt
    wt_change = round(latest_wt - initial_wt, 1) if initial_wt > 0 else 0.0
    return {
        'adherence_pct': adherence_pct,
        'completed_exercises': total_completed,
        'total_assigned': total_assigned,
        'avg_calories': avg_cals,
        'target_calories': target_cals,
        'initial_weight': initial_wt,
        'latest_weight': latest_wt,
        'weight_change': wt_change,
        'first_date': first_date.strftime('%b %d, %Y'),
        'latest_date': latest_date.strftime('%b %d, %Y'),
        'scheduled_rest_count': len(scheduled_rest_days),
        'additional_rest_count': additional_rest_days,
        'total_rest_count': len(scheduled_rest_days) + additional_rest_days,
        'workout_days_count': demographics.get('days_per_week', 4),
    }

def extract_timeline_series(user_data):
    weekly_logs = user_data.get('weekly_logs', {})
    sorted_weeks = []
    for week_key in weekly_logs:
        if week_key.startswith('week_'):
            try:
                w_date = datetime.date.fromisoformat(week_key.replace('week_', ''))
                sorted_weeks.append((w_date, week_key))
            except Exception:
                pass
    sorted_weeks.sort(key=lambda x: x[0])
    cal_points = []
    wt_points = []
    for monday_date, week_key in sorted_weeks:
        daily_metrics = weekly_logs[week_key].get('daily_metrics', {})
        for day_idx, day_name in enumerate(WEEKDAYS):
            cur_date = monday_date + datetime.timedelta(days=day_idx)
            day_label = cur_date.strftime('%b %d')
            day_data = daily_metrics.get(day_name, {})
            cals = day_data.get('calories', 0)
            wt = day_data.get('weight', 0.0)
            if cals and cals > 0:
                cal_points.append({'date': day_label, 'value': cals, 'day': day_name})
            if wt and wt > 0:
                wt_points.append({'date': day_label, 'value': wt, 'day': day_name})
    return cal_points, wt_points
