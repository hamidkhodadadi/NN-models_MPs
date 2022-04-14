
"""
Created on Sat Jan 23 19:42:26 2021

@author: ACER
"""

import tkinter as tk
import numpy as np
import torch

PATH1 = "Ra_deeper_entire_model.pt"
PATH2 = "Rb_deeper_entire_model.pt"
PATH3 = "Ca_deeper_entire_model_newFeatures.pt"
PATH4 = "Cb_deeper_entire_model_newFeatures.pt"

master = tk.Tk()
master.geometry("550x300")
master.title("Column Input Data") 
tk.Label(master, text="1. Enter Variables and Submit", width=45, anchor="w",font = "Times").grid(row=0)
tk.Label(master, text="Span/Depth ratio (a/d)", width=45, anchor="e").grid(row=1)
tk.Label(master, text="axial load ratio (P/Agf'c)", width=45, anchor="e").grid(row=2)
tk.Label(master, text="logitudinal reinforcement ratio (𝜌𝑙)", width=45, anchor="e").grid(row=3)
tk.Label(master, text="transverse reinforcement ratio (𝜌𝑡)", width=45, anchor="e").grid(row=4)
tk.Label(master, text="transverse rebar space over effective depth ratio (s/d)", width=45, anchor="e").grid(row=5)
tk.Label(master, text="shear load ratio (V𝑦/V0)", width=45, anchor="e").grid(row=6)
tk.Label(master, text="Compresive strength of the concrete (Mpa)", width=45, anchor="e").grid(row=7)
tk.Label(master, text="expected yeild strength of transverse reinforcement (Mpa)", width=45, anchor="e").grid(row=8)

e1 = tk.Entry(master, bd =2)
e2 = tk.Entry(master, bd =2)
e3 = tk.Entry(master, bd =2)
e4 = tk.Entry(master, bd =2)
e5 = tk.Entry(master, bd =2)
e6 = tk.Entry(master, bd =2)
e7 = tk.Entry(master, bd =2)
e8 = tk.Entry(master, bd =2)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)

def submit():
    global a,b,c,d,e,f,X,fc,fyt,Obs,ObsC,XC
    a = float(e1.get())
    b = float(e2.get())
    c = float(e3.get())
    d = float(e4.get())
    e = float(e5.get())
    f = float(e6.get())
    fc=float(e7.get())
    fyt=float(e8.get())
    X=np.array([a,b,c,d,e,f]).reshape(1,6)
    XC=np.array([a**2,a,b**2,b,c,d,e,f**2,f]).reshape(1,9)
    Obs=torch.Tensor(X).view(-1, 6) 
    ObsC=torch.Tensor(XC).view(-1, 9) 
    return a,b,c,d,e,f,fc,fyt,X,XC,Obs,ObsC

sub_btn=tk.Button(master,text = 'Submit', 
                  command = submit).grid(row=9, column=1, sticky=tk.W, pady=4)

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
    a_pred_NN=modela(ObsC).detach().numpy().reshape(len(XC))
    b_pred_NN=modelb(ObsC).detach().numpy().reshape(len(XC))
    a_ASCE=0.06 -0.06*(b)+1.3*(d)-0.037*(f)
    if b<=0.5:
       b1=5+(b/0.8)*(1/d)*(fc/fyt)
       b_ASCE=max((0.65/b1)-0.01,a_ASCE)
    elif b>=0.7:
         b_ASCE=0
    else:
        b1=5+(0.5/0.8)*(1/d)*(fc/fyt)
        ASCE_b1=(0.65/b1)-0.01
        b_ASCE=max(ASCE_b1*(1-((b-0.5)/0.2)),a_ASCE)
    c_ASCE=(0.24-0.4*(b))
    print('\n "Nueral Network model:"')
    print('nonlinear modeling parameter "a" is :',np.round(max(a_pred_NN,0),8))
    print('nonlinear modeling parameter "b" is : ',np.round(max(b_pred_NN,0),5))
    print()
    print('\n "ASCE41-17:"  "Columns not controlled by inadequate development or splicing along the clear height"')
    print('nonlinear modeling parameter "a"  is : ',np.round(max(a_ASCE,0),5))
    print('nonlinear modeling parameter "b"  is :',np.round(max(b_ASCE,0),5))
    print('residual strength ratio "c"  is : ',np.round(max(c_ASCE,0),5))
    print()
    C1=-0.65879653+(1.24815213*a)+(3.4648846*b)+(0.38969883*c)+(0.08088308*d)+(-2.14771188*e)+(-3.87871224*f)
    C2=4.11052344+(-1.0517519*a)+(-4.70313865*b)+(-0.35533031*c)+(-0.11361265*d)+(0.67877754*e)+(-1.58440226*f)
    C3=-3.36477767+(-1.81348808*a)+(0.74635459*b)+(-0.31776761*c)+(-0.1331492*d)+(4.56929014*e)+(4.86681085*f)
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
    a_ASCE=0.042 -0.043*(b)+0.63*(d)-0.023*(f)
    if b<=0.5:
       b1=5+(b/0.8)*(1/d)*(fc/fyt)
       b_ASCE=max((0.5/b1)-0.01,a_ASCE)
    elif b>=0.7:
         b_ASCE=0
    else:
        b1=5+(0.5/0.8)*(1/d)*(fc/fyt)
        ASCE_b1=(0.5/b1)-0.01
        b_ASCE=max(ASCE_b1*(1-((b-0.5)/0.2)),a_ASCE)
    c_ASCE=(0.24-0.4*(b))
    print('\n "Nueral Network model:"')    
    print('nonlinear modeling parameter "a"  is : ',np.round(max(a_pred_NN,0),5))
    print('nonlinear modeling parameter "b"  is : ',np.round(max(b_pred_NN,0),5))
    print()
    print('\n "ASCE41-17:"  "Columns not controlled by inadequate development or splicing along the clear height"')
    print('nonlinear modeling parameter "a"  is : ',np.round(max(a_ASCE,0),5))
    print('nonlinear modeling parameter "b"  is : ',np.round(max(b_ASCE,0),5))
    print('residual strength ratio "c"  is : ',np.round(max(c_ASCE,0),5))
    print()
    R1=2.42268724+(1.09516537*a)+(-0.93258533*b)+(0.29012392*c)+(0.20879405*d)+(-7.05765384*e)+(-4.75539328*f)
    R2= 0.31150907+(-0.72585348*a)+(-1.27997254*b)+(-0.21723133*c)+(-0.2752635*d)+(6.33579091*e)+(-1.65012577*f)
    R3=1.31196465+(-3.58821167*a)+(-0.23949947*b)+(0.08700118*c)+(-0.03825023*d)+(-1.04591491*e)+(7.28491747*f)
    if max(R1,R2,R3)==R1:
       print('\nPredicted Failure mode is "Flexure Critical (FC)"\n ')
    elif max(R1,R2,R3)==R2:
       print('\nPredicted Failure mode is "Flexure-Shear Critical (FSC)"\n ')
    else:
       print('\nPredicted Failure mode is "Shear Critical (SC)"\n ')
    #b_ASCE=
    

tk.Label(master, text="2. Choose Column Type", width=37, anchor="w",font = "Times").grid(row=10, column=0)
tk.Button(master, 
          text='Circular', 
          command=Cir, width=20).place(x=250, y=230)
tk.Button(master, text='Rectangular', command=Rec, width=20).place(x=250, y=260)

master.mainloop()