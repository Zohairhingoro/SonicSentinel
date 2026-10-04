from django.db import IntegrityError
from django.test import TestCase
from accounts.models import User


class UserModelTest(TestCase):
    def test_create_user_with_default_role_and_user_id(self):
        user = User.objects.create_user(username="testuser", password="password123")
        self.assertEqual(user.role, User.ROLE_NORMAL_USER)
        self.assertTrue(user.user_id.startswith("USR-"))
        self.assertTrue(user.is_normal_user)
        self.assertFalse(user.is_audio_reviewer)

    def test_user_roles(self):
        roles = [
            (User.ROLE_NORMAL_USER, "is_normal_user"),
            (User.ROLE_AUDIO_REVIEWER, "is_audio_reviewer"),
            (User.ROLE_SECURITY_OPERATOR, "is_security_operator"),
            (User.ROLE_MAINTENANCE_OPERATOR, "is_maintenance_operator"),
            (User.ROLE_ADMINISTRATOR, "is_administrator"),
        ]

        for role_val, prop_name in roles:
            user = User.objects.create_user(
                username=f"user_{role_val}",
                password="password123",
                role=role_val,
            )
            self.assertEqual(user.role, role_val)
            self.assertTrue(getattr(user, prop_name))

    def test_unique_user_id_constraint(self):
        user1 = User.objects.create_user(username="user1", password="password123")
        user2 = User.objects.create_user(username="user2", password="password123")
        self.assertNotEqual(user1.user_id, user2.user_id)

        with self.assertRaises(IntegrityError):
            User.objects.create_user(
                username="user3",
                password="password123",
                user_id=user1.user_id,
            )
