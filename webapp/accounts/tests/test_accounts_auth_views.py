from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

User = get_user_model()


class AuthViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()

        # Create Normal User
        self.normal_user = User.objects.create_user(
            username="normal_john",
            email="john@example.com",
            password="UserPassword123!",
            role=User.ROLE_NORMAL_USER,
        )

        # Create Reviewer User
        self.reviewer_user = User.objects.create_user(
            username="reviewer_mary",
            email="mary@example.com",
            password="UserPassword123!",
            role=User.ROLE_AUDIO_REVIEWER,
        )

        # Create Security Operator User
        self.operator_user = User.objects.create_user(
            username="operator_bob",
            email="bob@example.com",
            password="UserPassword123!",
            role=User.ROLE_SECURITY_OPERATOR,
        )

        # Create Administrator User
        self.admin_user = User.objects.create_user(
            username="admin_alice",
            email="alice@example.com",
            password="AdminPassword123!",
            role=User.ROLE_ADMINISTRATOR,
            is_staff=True,
            is_superuser=True,
        )

    def test_login_view_get(self):
        response = self.client.get(reverse("accounts:login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/login.html")

    def test_login_view_invalid_credentials(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "normal_john", "password": "WrongPassword!"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response.context["form"], None, "Please enter a correct username and password. Note that both fields may be case-sensitive.")

    def test_login_redirect_normal_user(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "normal_john", "password": "UserPassword123!"},
        )
        self.assertRedirects(response, reverse("dashboard:user"))

    def test_login_redirect_reviewer_operator(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "reviewer_mary", "password": "UserPassword123!"},
        )
        self.assertRedirects(response, reverse("dashboard:user"))

        self.client.logout()

        response2 = self.client.post(
            reverse("accounts:login"),
            {"username": "operator_bob", "password": "UserPassword123!"},
        )
        self.assertRedirects(response2, reverse("dashboard:user"))

    def test_login_redirect_administrator(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "admin_alice", "password": "AdminPassword123!"},
        )
        self.assertRedirects(response, reverse("dashboard:admin"))

    def test_logout_view(self):
        self.client.login(username="normal_john", password="UserPassword123!")
        response = self.client.post(reverse("accounts:logout"))
        self.assertRedirects(response, reverse("accounts:login"))
