# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:42:26 2021

@author: ACER
"""

import tkinter as tk
import numpy as np
import torch
import math

PATH1 = "Ra_deeper_entire_model.pt"
PATH2 = "Rb_deeper_entire_model.pt"
PATH3 = "Ca_deeper_entire_model_newFeatures.pt"
PATH4 = "Cb_deeper_entire_model_newFeatures.pt"

master = tk.Tk()
master.geometry("550x450")
master.title("Column Input Data") 
tk.Label(master, text="1. Enter input Variables and Submit", width=45, anchor="w",font = "Times").grid(row=0)
tk.Label(master, text="Gross column Area (Ag (mm2))", width=45, anchor="e").grid(row=1)
tk.Label(master, text="Section web width (bw (mm))", width=45, anchor="e").grid(row=2)

tk.Label(master, text="Shear Span length (a (mm))", width=45, anchor="e").grid(row=3)
tk.Label(master, text="Effective depth of the section (d (mm))", width=45, anchor="e").grid(row=4)
tk.Label(master, text="Column axial load (P (N))", width=45, anchor="e").grid(row=5)
tk.Label(master, text="Logitudinal reinforcement ratio (𝜌𝑙)", width=45, anchor="e").grid(row=6)
tk.Label(master, text="Transverse reinforcement ratio (𝜌𝑡)", width=45, anchor="e").grid(row=7)
tk.Label(master, text="Transverse rebar space(S (mm))", width=45, anchor="e").grid(row=8)
tk.Label(master, text="Yeilding moment capacity of the section (M𝑦 (N-mm))", width=45, anchor="e").grid(row=9)
tk.Label(master, text="Design bending  moment of the column (MUD(N-mm))", width=45, anchor="e").grid(row=10)
tk.Label(master, text="Design shear of the column (VUD(N))", width=45, anchor="e").grid(row=11)
#tk.Label(master, text="shear demand at flexural yeilding(V𝑦(N))", width=45, anchor="e").grid(row=6)
tk.Label(master, text="Compresive strength of the concrete (Mpa)", width=45, anchor="e").grid(row=12)
tk.Label(master, text="expected yeild strength of transverse reinforcement (Mpa)", width=45, anchor="e").grid(row=13)

e1 = tk.Entry(master, bd =2)
e2 = tk.Entry(master, bd =2)
e3 = tk.Entry(master, bd =2)
e4 = tk.Entry(master, bd =2)
e5 = tk.Entry(master, bd =2)
e6 = tk.Entry(master, bd =2)
e7 = tk.Entry(master, bd =2)
e8 = tk.Entry(master, bd =2)
e9 = tk.Entry(master, bd =2)
e10 = tk.Entry(master, bd =2)
e11 = tk.Entry(master, bd =2)
e12 = tk.Entry(master, bd =2)
e13 = tk.Entry(master, bd =2)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)

def submit():
    global x1,x2,x3,x4,x5,x6,VCol0,X,fc,fyt,Obs,ObsC
    Ag = float(e1.get())
    b = float(e2.get())
    a = float(e3.get())
    d = float(e4.get())
    p = float(e5.get())
    Rol = float(e6.get())
    Rot = float(e7.get())
    s = float(e8.get())
    My = float(e9.get())
    Mud = float(e10.get())
    Vud=float(e11.get())
    fc = float(e12.get())
    fyt = float(e13.get())
    x1=a/d
    x2=p/(Ag*fc)
    x3=Rol
    x4=Rot
    x5=s/d
    Vy=My/a
    Mratio=1
    if (s/d)<=0.75:
        alfa=1
    elif (s/d)>1:
        alfa=0
    else:
        alfa=-4*(s/d)+4
        
    #Mratio=(Mud/Vud/d)
        
    if (Mud/Vud/d)<=2:
        Mratio=2
    elif (Mud/Vud/d)>4:
        Mratio=4
    else:
        Mratio=(Mud/Vud/d)
    Av=Rot*b*s
    VCol0=alfa*(Av*fyt*d/s)+(0.5*math.sqrt(fc)/(Mratio)*math.sqrt(1+(p/(0.5*Ag*math.sqrt(fc))))*0.8*Ag)
    x6=Vy/VCol0
    X=np.array([x1,x2,x3,x4,x5,x6]).reshape(1,6)
    XC=np.array([x1**2,x1,x2**2,x2,x3,x4,x5,x6**2,x6]).T
    Obs=torch.Tensor(X).view(-1, 6)
    ObsC=torch.Tensor(XC).view(-1, 9) 
    Mratio=1*Mratio
    print('Mratio is :',Mratio)
    return Mratio,x1,x2,x3,x4,x5,x6,VCol0,fc,fyt,X,Obs

sub_btn=tk.Button(master,text = 'Submit', 
                  command = submit).grid(row=14, column=1, sticky=tk.W, pady=4)

#a = (float(e1.get()))
#b = (float(e2.get()))
#c = (float(e3.get()))
#d = (float(e4.get()))
#e = (float(e5.get()))
#f = (float(e6.get()))
#X=np.array([a,b,c,d,e,f]).reshape(1,6)
#Obs=torch.Tensor(X).view(-1, 6)  

def Cir():
    modela = torch.load(PATH3)
    modelb = torch.load(PATH4)
    global a_pred_NN
    global b_pred_NN
    a_pred_NN=modela(ObsC).detach().numpy().reshape(len(X))
    b_pred_NN=modelb(ObsC).detach().numpy().reshape(len(X))
    a_ASCE=0.06 -0.06*(x2)+1.3*(x4)-0.037*(x6)
    if x2<=0.5:
       b1=5+(x2/0.8)*(1/x4)*(fc/fyt)
       b_ASCE=max((0.65/b1)-0.01,a_ASCE)
    elif x2>=0.7:
         b_ASCE=0
    else:
        b1=5+(0.5/0.8)*(1/x4)*(fc/fyt)
        ASCE_b1=(0.65/b1)-0.01
        b_ASCE=max(ASCE_b1*(1-((x2-0.5)/0.2)),a_ASCE)
    c_ASCE=(0.24-0.4*(x2))
    print('\n "Nueral Network model:"')
    print('nonlinear modeling parameter "a" is :',np.round(max(a_pred_NN,0),8))
    print('nonlinear modeling parameter "b" is : ',np.round(max(b_pred_NN,0),5))
    print()
    print('\n "ASCE41-17:"  "Columns not controlled by inadequate development or splicing along the clear height"')
    print('nonlinear modeling parameter "a"  is : ',np.round(max(a_ASCE,0),5))
    print('nonlinear modeling parameter "b"  is :',np.round(max(b_ASCE,0),5))
    print('residual strength ratio "c"  is : ',np.round(max(c_ASCE,0),5))
    print()
    print('Shear capacity of the column (Vcol0) based on ASCE41-17 Eq. (10-3) is  : ',np.round(VCol0,3),'(N)')
    C1=-0.65879653+(1.24815213*x1)+(3.4648846*x2)+(0.38969883*x3)+(0.08088308*x4)+(-2.14771188*x5)+(-3.87871224*x6)
    C2=4.11052344+(-1.0517519*x1)+(-4.70313865*x2)+(-0.35533031*x3)+(-0.11361265*x4)+(0.67877754*x5)+(-1.58440226*x6)
    C3=-3.36477767+(-1.81348808*x1)+(0.74635459*x2)+(-0.31776761*x3)+(-0.1331492*x4)+(4.56929014*x5)+(4.86681085*x6)
    if max(C1,C2,C3)==C1:
       print('\nPredicted Failure mode is "Flexure Critical (FC)"\n ')
    elif max(C1,C2,C3)==C2:
       print('\nPredicted Failure mode is "Flexure-Shear Critical (FSC)"\n ')
    else:
       print('\nPredicted Failure mode is "Shear Critical (SC)"\n ')

def Rec():
    modela = torch.load(PATH1)
    modelb = torch.load(PATH2)
    global a_pred_NN
    global b_pred_NN
    a_pred_NN=modela(Obs).detach().numpy().reshape(len(X))
    b_pred_NN=modelb(Obs).detach().numpy().reshape(len(X))
    a_ASCE=0.042 -0.043*(x2)+0.63*(x4)-0.023*(x6)
    if x2<=0.5:
       b1=5+(x2/0.8)*(1/x4)*(fc/fyt)
       b_ASCE=max((0.5/b1)-0.01,a_ASCE)
    elif x2>=0.7:
         b_ASCE=0
    else:
        b1=5+(0.5/0.8)*(1/x4)*(fc/fyt)
        ASCE_b1=(0.5/b1)-0.01
        b_ASCE=max(ASCE_b1*(1-((x2-0.5)/0.2)),a_ASCE)
    c_ASCE=(0.24-0.4*(x2))
    print('\n "Nueral Network model:"')    
    print('nonlinear modeling parameter "a"  is : ',np.round(max(a_pred_NN,0),5))
    print('nonlinear modeling parameter "b"  is : ',np.round(max(b_pred_NN,0),5))
    print()
    print('\n "ASCE41-17:"  "Columns not controlled by inadequate development or splicing along the clear height"')
    print('nonlinear modeling parameter "a"  is : ',np.round(max(a_ASCE,0),5))
    print('nonlinear modeling parameter "b"  is : ',np.round(max(b_ASCE,0),5))
    print('residual strength ratio "c"  is : ',np.round(max(c_ASCE,0),5))
    print()
    print('Shear capacity of the column (Vcol0) based on ASCE41-17 Eq. (10-3) is  : ',np.round(VCol0,3),'(N)')

    R1=2.42268724+(1.09516537*x1)+(-0.93258533*x2)+(0.29012392*x3)+(0.20879405*x4)+(-7.05765384*x5)+(-4.75539328*x6)
    R2= 0.31150907+(-0.72585348*x1)+(-1.27997254*x2)+(-0.21723133*x3)+(-0.2752635*x4)+(6.33579091*x5)+(-1.65012577*x6)
    R3=1.31196465+(-3.58821167*x1)+(-0.23949947*x2)+(0.08700118*x3)+(-0.03825023*x4)+(-1.04591491*x5)+(7.28491747*x6)
    if max(R1,R2,R3)==R1:
       print('\nPredicted Failure mode is "Flexure Critical (FC)"\n ')
    elif max(R1,R2,R3)==R2:
       print('\nPredicted Failure mode is "Flexure-Shear Critical (FSC)"\n ')
    else:
       print('\nPredicted Failure mode is "Shear Critical (SC)"\n ')
    #b_ASCE=
    

tk.Label(master, text="2. Choose Column Type", width=37, anchor="w",font = "Times").grid(row=15, column=0)
tk.Button(master, 
          text='Circular', 
          command=Cir, width=20).place(x=250, y=350)
tk.Button(master, text='Rectangular', command=Rec, width=20).place(x=250, y=380)

master.mainloop()
