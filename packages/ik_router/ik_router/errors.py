"""Router errors.

ConfigurationError: missing API keys, missing service URLs, etc.
"""


class ConfigurationError(Exception):
    """Raised when a required configuration is missing."""
