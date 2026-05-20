from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set


ALLOWED_OPERATORS = {
    "abs",
    "add",
    "and",
    "bucket",
    "densify",
    "delta",
    "divide",
    "equal",
    "greater",
    "greater_equal",
    "group_backfill",
    "group_cartesian_product",
    "group_count",
    "group_mean",
    "group_neutralize",
    "group_rank",
    "group_scale",
    "group_std_dev",
    "group_sum",
    "group_zscore",
    "hump",
    "if_else",
    "inverse",
    "is_nan",
    "kth_element",
    "less",
    "less_equal",
    "log",
    "max",
    "min",
    "multiply",
    "normalize",
    "not",
    "not_equal",
    "or",
    "pasteurize",
    "power",
    "quantile",
    "rank",
    "reverse",
    "scale",
    "sign",
    "signed_power",
    "sqrt",
    "subtract",
    "tail",
    "trade_when",
    "ts_arg_max",
    "ts_arg_min",
    "ts_av_diff",
    "ts_backfill",
    "ts_corr",
    "ts_count_nans",
    "ts_covariance",
    "ts_decay_linear",
    "ts_delay",
    "ts_delta",
    "ts_ir",
    "ts_kurtosis",
    "ts_max_diff",
    "ts_mean",
    "ts_product",
    "ts_quantile",
    "ts_rank",
    "ts_regression",
    "ts_returns",
    "ts_scale",
    "ts_std_dev",
    "ts_step",
    "ts_sum",
    "ts_target_tvr_decay",
    "ts_target_tvr_hump",
    "ts_zscore",
    "winsorize",
    "zscore",
    "days_from_last_change",
    "last_diff_value",
    "vec_avg",
    "vec_count",
    "vec_max",
    "vec_min",
    "vec_range",
    "vec_stddev",
    "vec_sum",
}


BUILTIN_FIELD_IDENTIFIERS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "vwap",
    "cap",
    "adv20",
}

AUXILIARY_ONLY_FIELD_IDENTIFIERS = set(BUILTIN_FIELD_IDENTIFIERS)

GROUP_IDENTIFIERS = {
    "market",
    "sector",
    "industry",
    "subindustry",
    "country",
    "exchange",
}

RESERVED_IDENTIFIERS = {
    "true",
    "false",
    "nan",
    "none",
    "std",
    "rate",
    "dense",
    "constant",
    "filter",
    "lag",
    "rettype",
}

EXACT_OPERATOR_ARITY = {
    "group_mean": 3,
    "hump": 1,
}


def validate_expression(
    expression: str,
    max_operator_count: int = 14,
    max_length: int = 1200,
    allowed_fields: Iterable[str] | None = None,
    field_types: Dict[str, str] | None = None,
    enforce_auxiliary_field_roles: bool = False,
    auxiliary_fields: Iterable[str] | None = None,
) -> List[str]:
    errors: List[str] = []
    text = str(expression or "").strip()
    if not text:
        return ["EMPTY_EXPRESSION"]
    if len(text) > max_length:
        errors.append(f"EXPRESSION_TOO_LONG:{len(text)}>{max_length}")
    if not _balanced_parentheses(text):
        errors.append("UNBALANCED_PARENTHESES")
    if _has_empty_function_argument(text):
        errors.append("EMPTY_FUNCTION_ARGUMENT")

    operators = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if len(operators) > max_operator_count:
        errors.append(f"TOO_MANY_OPERATORS:{len(operators)}>{max_operator_count}")
    for op in operators:
        if op not in ALLOWED_OPERATORS:
            errors.append(f"UNKNOWN_OPERATOR:{op}")
    if allowed_fields:
        allowed = _field_allowlist(allowed_fields)
        for field in _field_identifiers(text, operators):
            if field not in allowed and field.lower() not in allowed:
                errors.append(f"UNKNOWN_FIELD:{field}")
    if enforce_auxiliary_field_roles:
        errors.extend(_auxiliary_primary_field_errors(text, auxiliary_fields))
    errors.extend(_operator_arity_errors(text))
    errors.extend(_vector_reducer_arity_errors(text))
    if field_types:
        errors.extend(_vector_reducer_type_errors(text, field_types))
        errors.extend(_vector_time_series_errors(text, field_types))
    return errors


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_empty_function_argument(text: str) -> bool:
    return bool(re.search(r"\(\s*,|,\s*,|,\s*\)", text))


def _field_allowlist(allowed_fields: Iterable[str]) -> Set[str]:
    values = {str(field).strip() for field in allowed_fields if str(field).strip()}
    values.update(BUILTIN_FIELD_IDENTIFIERS)
    return values | {field.lower() for field in values}


def _field_identifiers(text: str, operators: List[str]) -> List[str]:
    operator_set = set(operators)
    identifiers = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text):
        name = match.group(1)
        tail = text[match.end() :]
        next_nonspace = tail.lstrip()[:1]
        if name in operator_set or name in ALLOWED_OPERATORS:
            continue
        if name.lower() in BUILTIN_FIELD_IDENTIFIERS | GROUP_IDENTIFIERS | RESERVED_IDENTIFIERS:
            continue
        if next_nonspace in {"(", "="}:
            continue
        identifiers.append(name)
    return list(dict.fromkeys(identifiers))


def _all_field_identifiers(text: str) -> List[str]:
    identifiers = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text):
        name = match.group(1)
        tail = text[match.end() :]
        next_nonspace = tail.lstrip()[:1]
        if name in ALLOWED_OPERATORS:
            continue
        if name.lower() in GROUP_IDENTIFIERS | RESERVED_IDENTIFIERS:
            continue
        if next_nonspace in {"(", "="}:
            continue
        identifiers.append(name)
    return list(dict.fromkeys(identifiers))


def _auxiliary_primary_field_errors(text: str, auxiliary_fields: Iterable[str] | None) -> List[str]:
    auxiliary = _normalized_auxiliary_fields(auxiliary_fields)
    if not auxiliary:
        return []
    if not _auxiliary_primary_violation(text, auxiliary):
        return []
    fields = [
        field
        for field in _all_field_identifiers(text)
        if field.lower() in auxiliary
    ]
    return [f"AUXILIARY_FIELD_AS_PRIMARY:{','.join(_sorted_unique_fields(fields))}"]


def _normalized_auxiliary_fields(auxiliary_fields: Iterable[str] | None) -> Set[str]:
    fields = auxiliary_fields or AUXILIARY_ONLY_FIELD_IDENTIFIERS
    return {str(field).strip().lower() for field in fields if str(field).strip()}


def _auxiliary_primary_violation(text: str, auxiliary_fields: Set[str]) -> bool:
    text = _strip_outer_parentheses(str(text or "").strip())
    if not text:
        return False
    if _is_auxiliary_only_expression(text, auxiliary_fields):
        return True

    root = _root_function(text)
    if root is None:
        return False
    operator, args = root
    if not args:
        return False

    if operator in _PRIMARY_SIGNAL_WRAPPERS:
        return _auxiliary_primary_violation(args[0], auxiliary_fields)
    if operator in {"add", "subtract", "max", "min"}:
        return any(
            _is_auxiliary_only_expression(arg, auxiliary_fields)
            or _auxiliary_primary_violation(arg, auxiliary_fields)
            for arg in args
        )
    if operator == "divide":
        numerator = args[0]
        return _is_auxiliary_only_expression(numerator, auxiliary_fields) or _auxiliary_primary_violation(
            numerator,
            auxiliary_fields,
        )
    if operator == "if_else":
        return any(
            _is_auxiliary_only_expression(arg, auxiliary_fields)
            or _auxiliary_primary_violation(arg, auxiliary_fields)
            for arg in args[1:3]
        )
    if operator == "trade_when" and len(args) >= 2:
        alpha_arg = args[1]
        return _is_auxiliary_only_expression(alpha_arg, auxiliary_fields) or _auxiliary_primary_violation(
            alpha_arg,
            auxiliary_fields,
        )
    return False


_PRIMARY_SIGNAL_WRAPPERS = {
    "abs",
    "group_neutralize",
    "group_rank",
    "group_scale",
    "group_zscore",
    "hump",
    "inverse",
    "log",
    "normalize",
    "pasteurize",
    "quantile",
    "rank",
    "reverse",
    "scale",
    "sign",
    "signed_power",
    "sqrt",
    "ts_arg_max",
    "ts_arg_min",
    "ts_av_diff",
    "ts_backfill",
    "ts_count_nans",
    "ts_decay_linear",
    "ts_delay",
    "ts_delta",
    "ts_ir",
    "ts_kurtosis",
    "ts_max_diff",
    "ts_mean",
    "ts_product",
    "ts_quantile",
    "ts_rank",
    "ts_returns",
    "ts_scale",
    "ts_std_dev",
    "ts_sum",
    "ts_target_tvr_decay",
    "ts_target_tvr_hump",
    "ts_zscore",
    "winsorize",
    "zscore",
}


def _is_auxiliary_only_expression(text: str, auxiliary_fields: Set[str]) -> bool:
    fields = _all_field_identifiers(text)
    return bool(fields) and all(field.lower() in auxiliary_fields for field in fields)


def _root_function(text: str) -> tuple[str, List[str]] | None:
    text = _strip_outer_parentheses(text)
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if not match:
        return None
    open_paren_index = text.find("(", match.start(1))
    close_paren_index = _matching_paren_index(text, open_paren_index)
    if close_paren_index is None or text[close_paren_index + 1 :].strip():
        return None
    args = _function_arguments(text, open_paren_index)
    if args is None:
        return None
    return match.group(1), args


def _matching_paren_index(text: str, open_paren_index: int) -> int | None:
    depth = 0
    for index in range(open_paren_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def _strip_outer_parentheses(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        close_index = _matching_paren_index(text, 0)
        if close_index != len(text) - 1:
            break
        text = text[1:-1].strip()
    return text


def _sorted_unique_fields(fields: List[str]) -> List[str]:
    seen = set()
    values = []
    for field in fields:
        normalized = field.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return sorted(values)


def _vector_time_series_errors(text: str, field_types: Dict[str, str]) -> List[str]:
    type_map = _normalized_field_types(field_types)
    vector_fields = {field for field, field_type in type_map.items() if field_type == "VECTOR"}
    vector_fields.update({field.lower() for field in vector_fields})
    errors: List[str] = []
    pattern = r"(?=\b(ts_[A-Za-z0-9_]+)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\b)"
    for match in re.finditer(pattern, text):
        operator = match.group(1)
        first_arg = match.group(2)
        if first_arg in vector_fields or first_arg.lower() in vector_fields:
            errors.append(f"INVALID_VECTOR_TS_OPERATOR:{operator}:{first_arg}")
    return list(dict.fromkeys(errors))


def _vector_reducer_type_errors(text: str, field_types: Dict[str, str]) -> List[str]:
    type_map = _normalized_field_types(field_types)
    errors: List[str] = []
    for match in re.finditer(r"\b(vec_(?:avg|count|max|min|range|stddev|sum))\s*\(", text):
        operator = match.group(1)
        args = _function_arguments(text, match.end() - 1)
        if args is None or len(args) != 1:
            continue
        argument = args[0].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", argument):
            continue
        field_type = type_map.get(argument) or type_map.get(argument.lower())
        if field_type and field_type != "VECTOR":
            errors.append(f"INVALID_VECTOR_REDUCER_INPUT_TYPE:{operator}:{argument}:{field_type}")
    return list(dict.fromkeys(errors))


def _vector_reducer_arity_errors(text: str) -> List[str]:
    errors: List[str] = []
    for match in re.finditer(r"\b(vec_(?:avg|count|max|min|range|stddev|sum))\s*\(", text):
        operator = match.group(1)
        args = _function_arguments(text, match.end() - 1)
        if args is None:
            continue
        if len(args) != 1:
            errors.append(f"INVALID_VECTOR_REDUCER_ARITY:{operator}")
    return list(dict.fromkeys(errors))


def _operator_arity_errors(text: str) -> List[str]:
    errors: List[str] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        operator = match.group(1)
        expected = EXACT_OPERATOR_ARITY.get(operator)
        if expected is None:
            continue
        args = _function_arguments(text, match.end() - 1)
        if args is None:
            continue
        actual = len(args)
        if actual != expected:
            errors.append(f"INVALID_OPERATOR_ARITY:{operator}:{actual}!={expected}")
    return list(dict.fromkeys(errors))


def _normalized_field_types(field_types: Dict[str, str]) -> Dict[str, str]:
    normalized = {}
    for field, field_type in field_types.items():
        name = str(field).strip()
        if not name:
            continue
        normalized[name] = str(field_type).strip().upper()
        normalized[name.lower()] = normalized[name]
    return normalized


def _function_arguments(text: str, open_paren_index: int) -> List[str] | None:
    depth = 0
    current = []
    args: List[str] = []
    started = False
    for index in range(open_paren_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
            if started:
                current.append(char)
            started = True
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0:
                arg = "".join(current).strip()
                if arg or args:
                    args.append(arg)
                return args
            current.append(char)
            continue
        if char == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
            continue
        if started:
            current.append(char)
    return None
