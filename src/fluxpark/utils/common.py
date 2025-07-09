from string import Formatter


def has_placeholders(pattern: str) -> bool:
    """Return True als pattern één of meer {veld}-placeholders bevat."""
    for _, field_name, _, _ in Formatter().parse(pattern):
        if field_name:
            return True
    return False
