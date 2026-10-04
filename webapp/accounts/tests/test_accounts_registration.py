from django.contrib.auth import get_user_model
from django.contrib.messages import get_messages
from django.test import TestCase
from django.urls import reverse

from accounts.forms import UserRegistrationForm

User = get_user_model()


class UserRegistrationTestCase(TestCase):
    def test_form_allowed_roles(self):
        # Valid non-admin roles
        for role in [
            User.ROLE_NORMAL_USER,
            User.ROLE_AUDIO_REVIEWER,
            User.ROLE_SECURITY_OPERATOR,
            User.ROLE_MAINTENANCE_OPERATOR,
        ]:
            form_data = {
                "username": f"user_{role}",
                "email": f"{role}@example.com",
                "role": role,
                "password1": "SecurePass123!#$",
                "password2": "SecurePass123!#$",
            }
            form = UserRegistrationForm(data=form_data)
            self.assertTrue(form.is_valid(), f"Form should be valid for role {role}: {form.errors}")

    def test_form_rejects_administrator_role(self):
        form_data = {
            "username": "hacker_admin",
            "email": "hacker@example.com",
            "role": User.ROLE_ADMINISTRATOR,
            "password1": "SecurePass123!#$",
            "password2": "SecurePass123!#$",
        }
        form = UserRegistrationForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("role", form.errors)

    def test_register_view_get(self):
        response = self.client.get(reverse("accounts:register"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/register.html")
        self.assertIn("form", response.context)

    def test_register_view_post_success(self):
        post_data = {
            "username": "new_reviewer",
            "email": "reviewer@example.com",
            "role": User.ROLE_AUDIO_REVIEWER,
            "password1": "ComplexPass123!#$",
            "password2": "ComplexPass123!#$",
        }
        response = self.client.post(reverse("accounts:register"), post_data)
        self.assertRedirects(response, reverse("accounts:login"))

        # Verify user created in DB with correct role
        user = User.objects.get(username="new_reviewer")
        self.assertEqual(user.email, "reviewer@example.com")
        self.assertEqual(user.role, User.ROLE_AUDIO_REVIEWER)
        self.assertTrue(user.is_audio_reviewer)
        self.assertTrue(user.user_id.startswith("USR-"))

        # Verify success message
        messages = list(get_messages(response.wsgi_request))
        self.assertEqual(len(messages), 1)
        self.assertIn("Account created successfully", str(messages[0]))

    def test_register_view_post_validation_error(self):
        post_data = {
            "username": "short_pass_user",
            "email": "short@example.com",
            "role": User.ROLE_NORMAL_USER,
            "password1": "123",
            "password2": "123",
        }
        response = self.client.post(reverse("accounts:register"), post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="short_pass_user").exists())
