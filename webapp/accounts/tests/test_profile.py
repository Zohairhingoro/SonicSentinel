from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()


class ProfileViewTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="john_doe",
            email="john@example.com",
            password="Password123!",
            first_name="John",
            last_name="Doe",
            role=User.ROLE_AUDIO_REVIEWER,
        )
        self.other_user = User.objects.create_user(
            username="jane_doe",
            email="jane@example.com",
            password="Password123!",
        )

    def test_unauthenticated_profile_redirects_to_login(self):
        response = self.client.get(reverse("accounts:profile"))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("accounts:login"), response.url)

    def test_authenticated_profile_get(self):
        self.client.login(username="john_doe", password="Password123!")
        response = self.client.get(reverse("accounts:profile"))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/profile.html")
        self.assertContains(response, self.user.user_id)
        self.assertContains(response, "Audio Reviewer")
        self.assertContains(response, "john@example.com")

    def test_authenticated_profile_post_update_success(self):
        self.client.login(username="john_doe", password="Password123!")
        response = self.client.post(
            reverse("accounts:profile"),
            {
                "first_name": "Johnny",
                "last_name": "Smith",
                "email": "johnny_updated@example.com",
            },
        )

        self.assertRedirects(response, reverse("accounts:profile"))

        self.user.refresh_from_db()
        self.assertEqual(self.user.first_name, "Johnny")
        self.assertEqual(self.user.last_name, "Smith")
        self.assertEqual(self.user.email, "johnny_updated@example.com")
        self.assertEqual(self.user.role, User.ROLE_AUDIO_REVIEWER)  # Role unchanged

    def test_authenticated_profile_post_duplicate_email_fails(self):
        self.client.login(username="john_doe", password="Password123!")
        response = self.client.post(
            reverse("accounts:profile"),
            {
                "first_name": "John",
                "last_name": "Doe",
                "email": "jane@example.com",  # Email already taken by other_user
            },
        )

        self.assertEqual(response.status_code, 200)
        form = response.context["form"]
        self.assertIn("email", form.errors)
        self.assertIn("This email address is already registered to another user.", form.errors["email"])
        self.user.refresh_from_db()
        self.assertEqual(self.user.email, "john@example.com")
