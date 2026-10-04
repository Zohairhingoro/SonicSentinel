"""
tests/django/test_user_model.py — Unit tests for SonicSentinel AI custom User model.

Tests:
  - Role assignment for all 5 mandatory roles (normal_user, audio_reviewer,
    security_operator, maintenance_operator, administrator)
  - Role helper properties (is_normal_user, is_audio_reviewer, is_security_operator,
    is_maintenance_operator, is_administrator) and mutual exclusivity
  - Superuser role inheritance (is_administrator returns True for superusers)
  - Unique User ID generation, format (USR-XXXXXXXXXXXX), uniqueness, DB constraint,
    and preservation across updates
"""

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

User = get_user_model()


class UserModelRoleAssignmentTestCase(TestCase):
    """Verifies role assignment and role helper properties on the custom User model."""

    def test_default_role_is_normal_user(self):
        """A new user created without an explicit role defaults to normal_user."""
        user = User.objects.create_user(username="default_user", password="secure_password123")
        self.assertEqual(user.role, User.ROLE_NORMAL_USER)
        self.assertTrue(user.is_normal_user)
        self.assertFalse(user.is_audio_reviewer)
        self.assertFalse(user.is_security_operator)
        self.assertFalse(user.is_maintenance_operator)
        self.assertFalse(user.is_administrator)

    def test_role_choices_contain_all_five_roles(self):
        """The User model must declare exactly the 5 mandatory system roles."""
        expected_roles = {
            User.ROLE_NORMAL_USER,
            User.ROLE_AUDIO_REVIEWER,
            User.ROLE_SECURITY_OPERATOR,
            User.ROLE_MAINTENANCE_OPERATOR,
            User.ROLE_ADMINISTRATOR,
        }
        defined_roles = {choice[0] for choice in User.ROLE_CHOICES}
        self.assertEqual(defined_roles, expected_roles)

    def test_role_assignment_normal_user(self):
        """User assigned normal_user role has correct properties."""
        user = User.objects.create_user(
            username="normal_user_1", password="password123", role=User.ROLE_NORMAL_USER
        )
        self.assertEqual(user.role, User.ROLE_NORMAL_USER)
        self.assertTrue(user.is_normal_user)
        self.assertFalse(user.is_audio_reviewer)
        self.assertFalse(user.is_security_operator)
        self.assertFalse(user.is_maintenance_operator)
        self.assertFalse(user.is_administrator)

    def test_role_assignment_audio_reviewer(self):
        """User assigned audio_reviewer role has correct properties."""
        user = User.objects.create_user(
            username="reviewer_1", password="password123", role=User.ROLE_AUDIO_REVIEWER
        )
        self.assertEqual(user.role, User.ROLE_AUDIO_REVIEWER)
        self.assertTrue(user.is_audio_reviewer)
        self.assertFalse(user.is_normal_user)
        self.assertFalse(user.is_security_operator)
        self.assertFalse(user.is_maintenance_operator)
        self.assertFalse(user.is_administrator)

    def test_role_assignment_security_operator(self):
        """User assigned security_operator role has correct properties."""
        user = User.objects.create_user(
            username="sec_op_1", password="password123", role=User.ROLE_SECURITY_OPERATOR
        )
        self.assertEqual(user.role, User.ROLE_SECURITY_OPERATOR)
        self.assertTrue(user.is_security_operator)
        self.assertFalse(user.is_normal_user)
        self.assertFalse(user.is_audio_reviewer)
        self.assertFalse(user.is_maintenance_operator)
        self.assertFalse(user.is_administrator)

    def test_role_assignment_maintenance_operator(self):
        """User assigned maintenance_operator role has correct properties."""
        user = User.objects.create_user(
            username="maint_op_1", password="password123", role=User.ROLE_MAINTENANCE_OPERATOR
        )
        self.assertEqual(user.role, User.ROLE_MAINTENANCE_OPERATOR)
        self.assertTrue(user.is_maintenance_operator)
        self.assertFalse(user.is_normal_user)
        self.assertFalse(user.is_audio_reviewer)
        self.assertFalse(user.is_security_operator)
        self.assertFalse(user.is_administrator)

    def test_role_assignment_administrator(self):
        """User assigned administrator role has correct properties."""
        user = User.objects.create_user(
            username="admin_1", password="password123", role=User.ROLE_ADMINISTRATOR
        )
        self.assertEqual(user.role, User.ROLE_ADMINISTRATOR)
        self.assertTrue(user.is_administrator)
        self.assertFalse(user.is_normal_user)
        self.assertFalse(user.is_audio_reviewer)
        self.assertFalse(user.is_security_operator)
        self.assertFalse(user.is_maintenance_operator)

    def test_superuser_is_administrator(self):
        """Superuser without explicit admin role still evaluates as administrator."""
        superuser = User.objects.create_superuser(
            username="root_admin", password="password123", role=User.ROLE_NORMAL_USER
        )
        self.assertTrue(superuser.is_superuser)
        self.assertTrue(superuser.is_administrator)

    def test_string_representation(self):
        """__str__ includes the username and role for operational logging."""
        user = User.objects.create_user(
            username="analyst_jane", password="password123", role=User.ROLE_AUDIO_REVIEWER
        )
        self.assertEqual(str(user), "analyst_jane (audio_reviewer)")


class UserModelUniqueUserIdTestCase(TestCase):
    """Verifies unique User ID auto-generation, format, uniqueness, and constraints."""

    def test_user_id_auto_generated_on_creation(self):
        """User ID is populated automatically when a user is saved without one."""
        user = User.objects.create_user(username="user_autoid", password="password123")
        self.assertIsNotNone(user.user_id)
        self.assertTrue(len(user.user_id) > 4)

    def test_user_id_format(self):
        """User ID starts with 'USR-' followed by 12 uppercase hexadecimal characters."""
        user = User.objects.create_user(username="user_fmt", password="password123")
        self.assertTrue(user.user_id.startswith("USR-"))
        self.assertEqual(len(user.user_id), 16)
        hex_part = user.user_id[4:]
        self.assertTrue(all(c in "0123456789ABCDEF" for c in hex_part))

    def test_user_id_uniqueness(self):
        """Each created user receives a distinct User ID."""
        user1 = User.objects.create_user(username="user_unique_1", password="password123")
        user2 = User.objects.create_user(username="user_unique_2", password="password123")
        user3 = User.objects.create_user(username="user_unique_3", password="password123")
        ids = {user1.user_id, user2.user_id, user3.user_id}
        self.assertEqual(len(ids), 3)

    def test_unique_user_id_constraint_enforced_by_db(self):
        """Attempting to save a user with an already existing user_id raises IntegrityError."""
        user1 = User.objects.create_user(username="user_orig", password="password123")
        with self.assertRaises(IntegrityError):
            User.objects.create_user(
                username="user_duplicate",
                password="password123",
                user_id=user1.user_id,
            )

    def test_user_id_preserved_on_update(self):
        """Modifying other user attributes does not regenerate or mutate user_id."""
        user = User.objects.create_user(
            username="user_mutate", password="password123", role=User.ROLE_NORMAL_USER
        )
        initial_id = user.user_id
        user.email = "updated@example.com"
        user.first_name = "Jane"
        user.role = User.ROLE_AUDIO_REVIEWER
        user.save()

        user.refresh_from_db()
        self.assertEqual(user.user_id, initial_id)
        self.assertEqual(user.email, "updated@example.com")
        self.assertEqual(user.role, User.ROLE_AUDIO_REVIEWER)

    def test_explicit_user_id_preserved(self):
        """If user_id is explicitly provided at creation, it is respected and not overwritten."""
        custom_id = "USR-CUSTOM000001"
        user = User.objects.create_user(
            username="user_custom", password="password123", user_id=custom_id
        )
        self.assertEqual(user.user_id, custom_id)
        user.refresh_from_db()
        self.assertEqual(user.user_id, custom_id)
