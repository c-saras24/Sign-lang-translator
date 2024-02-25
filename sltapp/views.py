from django.shortcuts import render,redirect
from .models import *
from .forms import *
import os
import sys
import subprocess
from django.contrib.auth.models import Group
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required,user_passes_test
from django.core.files.storage import FileSystemStorage

# Create your views here.


def index(request):
    return render(request,'index.html')

def register(request):
    form=dform()
    f2=cform()
    mydict={'form':form,'f2':f2}
    if request.method=="POST":
        form=dform(request.POST)
        f2=cform(request.POST)
        if form.is_valid() and f2.is_valid():
            user=form.save()
            user.set_password(user.password)
            user.save()
            cmodel=f2.save(commit=True)
            cmodel.user=user
            cmodel.save()
            group=Group.objects.get_or_create(name="userpage")
            group[0].user_set.add(user)
            return redirect('login')
    return render(request,'register.html',mydict)

def is_cmodel(user):
    return user.groups.filter(name="userpage").exists()

def login(request):
    return render(request,'login.html')

def afterlogin_view(request):
    if is_cmodel(request.user):
        return redirect('dashb')




@login_required(login_url="login")
@user_passes_test(is_cmodel)
def dashb(request):
    return render(request,'dashb.html')

def file(request):
    command='python C:\\Users\\user\\OneDrive\\Desktop\\slt\\sltapp\\SignDetector.py'
    os.system(command)
    return render(request,'file.html')     


@login_required(login_url="login")
@user_passes_test(is_cmodel)
def profile(request):
    pf=cmodel.objects.get(user_id=request.user.id)
    if request.method == "POST":
        pf=cmodel.objects.get(user_id=request.user)
        dpro=dform(request.POST,instance=pf.user)
        cpro=cform(request.POST,instance=pf)
        if dpro.is_valid() and cpro.is_valid():
            user=dpro.save()
            user.set_password(user.password)
            user.save()
            cpro.save()
            return redirect('login')
    else:
        dpro=dform(instance=pf.user)
        cpro=cform(instance=pf)
        context={
            'dpro':dpro,
            'cpro':cpro,
        }
    return render(request,'profile.html',context)

@login_required
def logout_request(request):
    logout(request)
    return redirect('/')
