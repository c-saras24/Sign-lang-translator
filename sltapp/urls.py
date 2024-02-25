from django.contrib.auth.views import LoginView
from django.urls import path
from .import views

urlpatterns=[
    path('',views.index,name="index"),
    path('index/',views.index,name="index"),
    path('index/',views.index,name="index"),
    path('register/',views.register,name="register"),
    path('login/',LoginView.as_view(template_name='login.html'),name="login"),
    path('afterlogin',views.afterlogin_view,name="afterlogin"),
    path('dashb/',views.dashb,name="dashb"),
    path('profile/',views.profile,name="profile"),
    path('file/',views.file,name="file"),
    path('logout/',views.logout_request,name="logout")

]