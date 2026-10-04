from django import template
from accounts.permissions import (
    is_administrator,
    is_audio_reviewer,
    is_maintenance_operator,
    is_normal_user,
    is_security_operator,
    is_staff_user,
)

register = template.Library()


@register.filter(name="is_normal_user")
def is_normal_user_filter(user):
    return is_normal_user(user)


@register.filter(name="is_audio_reviewer")
def is_audio_reviewer_filter(user):
    return is_audio_reviewer(user)


@register.filter(name="is_security_operator")
def is_security_operator_filter(user):
    return is_security_operator(user)


@register.filter(name="is_maintenance_operator")
def is_maintenance_operator_filter(user):
    return is_maintenance_operator(user)


@register.filter(name="is_administrator")
def is_administrator_filter(user):
    return is_administrator(user)


@register.filter(name="is_staff_user")
def is_staff_user_filter(user):
    return is_staff_user(user)
