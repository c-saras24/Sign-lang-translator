from django import forms
from .import models
from django.contrib.auth.models import User


class dform(forms.ModelForm):
    class Meta:
        model=User
        fields=['username','password','email']
        widgets={
            'password': forms.PasswordInput()

        }

class cform(forms.ModelForm):
    class Meta:
        model=models.cmodel
        fields=['mobile']