"""
String utility operations: case conversion, text format helpers.
"""
import re


def camel_case_to_snake_case(string: str) -> str:
    """
    Convert a CamelCase string to snake_case.
    
    :param string: Input CamelCase string
    :return: Lowercase, underscore-separated string
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()
