from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

User = get_user_model()


class UserRegistrationForm(UserCreationForm):
    """User registration form allowing choice of non-administrator roles."""

    ALLOWED_ROLES = [
        (User.ROLE_NORMAL_USER, "Normal User"),
        (User.ROLE_AUDIO_REVIEWER, "Audio Reviewer"),
        (User.ROLE_SECURITY_OPERATOR, "Security Operator"),
        (User.ROLE_MAINTENANCE_OPERATOR, "Maintenance Operator"),
    ]

    email = forms.EmailField(
        required=True,
        label="Email Address",
        help_text="Required for notifications and identification.",
        widget=forms.EmailInput(
            attrs={
                "class": "form-control",
                "placeholder": "user@example.com",
            }
        ),
    )

    role = forms.ChoiceField(
        choices=ALLOWED_ROLES,
        required=True,
        initial=User.ROLE_NORMAL_USER,
        label="Operational Role",
        help_text="Select your role. Administrator accounts are assigned by system management.",
        widget=forms.Select(
            attrs={
                "class": "form-select",
            }
        ),
    )

    class Meta:
        model = User
        fields = ("username", "email", "role")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Apply component form-control class to username, password1, password2 fields
        field_attrs = {"class": "form-control"}
        for field in ("username", "password1", "password2"):
            if field in self.fields:
                self.fields[field].widget.attrs.update(field_attrs)
        if "password1" in self.fields:
            self.fields["password1"].help_text = None


class UserProfileForm(forms.ModelForm):
    """Form allowing logged-in user to update profile details (name and email)."""

    first_name = forms.CharField(
        required=False,
        label="First Name",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "First Name",
            }
        ),
    )

    last_name = forms.CharField(
        required=False,
        label="Last Name",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Last Name",
            }
        ),
    )

    email = forms.EmailField(
        required=True,
        label="Email Address",
        widget=forms.EmailInput(
            attrs={
                "class": "form-control",
                "placeholder": "user@example.com",
            }
        ),
    )

    class Meta:
        model = User
        fields = ("first_name", "last_name", "email")

    def clean_email(self) -> str:
        email = self.cleaned_data.get("email", "").strip()
        if User.objects.exclude(pk=self.instance.pk).filter(email__iexact=email).exists():
            raise forms.ValidationError("This email address is already registered to another user.")
        return email


