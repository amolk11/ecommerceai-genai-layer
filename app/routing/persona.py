"""Persona definitions for the GenAI Layer."""

from enum import Enum


class Persona(str, Enum):
    """Supported user personas in the GenAI Layer."""

    BUSINESS = "business"
    CUSTOMER = "customer"
    DEVELOPER = "developer"
    