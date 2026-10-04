from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView as DjangoLoginView, LogoutView as DjangoLogoutView
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views import View

from .forms import UserProfileForm, UserRegistrationForm
from .permissions import require_staff

User = get_user_model()


@require_staff
def index(request):
    """Example function-based view demonstrating require_staff decorator."""
    return JsonResponse({"status": "ok", "app": "accounts"})


class ProfileView(LoginRequiredMixin, View):
    """View allowing logged-in user to view and update profile information."""

    login_url = reverse_lazy("accounts:login")

    def get(self, request):
        form = UserProfileForm(instance=request.user)
        return render(
            request,
            "accounts/profile.html",
            {
                "form": form,
                "user_obj": request.user,
            },
        )

    def post(self, request):
        form = UserProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile information has been updated successfully.")
            return redirect("accounts:profile")
        return render(
            request,
            "accounts/profile.html",
            {
                "form": form,
                "user_obj": request.user,
            },
        )



class RegisterView(View):
    """View handling user registration with role selection."""

    def get(self, request):
        if request.user.is_authenticated:
            return redirect("home")
        form = UserRegistrationForm()
        return render(request, "accounts/register.html", {"form": form})

    def post(self, request):
        if request.user.is_authenticated:
            return redirect("home")
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(
                request,
                f"Account created successfully for {user.username}. You can now log in.",
            )
            return redirect("accounts:login")
        return render(request, "accounts/register.html", {"form": form})


class UserLoginView(DjangoLoginView):
    """View handling user authentication using Django's default auth system."""

    template_name = "accounts/login.html"
    redirect_authenticated_user = True

    def get_success_url(self) -> str:
        """Redirect based on user role: Administrator -> dashboard:admin, others -> dashboard:user."""
        user = self.request.user
        if (
            getattr(user, "is_administrator", False)
            or getattr(user, "is_superuser", False)
            or getattr(user, "role", "") == User.ROLE_ADMINISTRATOR
        ):
            return str(reverse_lazy("dashboard:admin"))
        return str(reverse_lazy("dashboard:user"))


class UserLogoutView(DjangoLogoutView):
    """View handling user logout using Django's default auth system."""

    next_page = reverse_lazy("accounts:login")
