#!usr/bin/python
#coding: utf-8
import math
import random
from numpy import*
"""
本文主要是对于简单的BP神经网络的实现
"""

def sigmond(x):
    ans=1.0/(1.0+exp(-1.0*x))
    return ans

def rand_bet(a,b):
    ans=random.random()*(b-a)+a
    return ans

class BP_net:
    def __init__(self,ni,nh,no):
        self.ni=ni
        self.nh=nh
        self.no=no

        self.ai=zeros(ni)
        self.ah=zeros(nh)
        self.ao=zeros(no)

        self.wei_ih = zeros([ni, nh])
        self.wei_ho = zeros([nh, no])
        for i in range(ni):
            for j in range(nh):
                self.wei_ih[i][j]=rand_bet(-0.2,0.2)
        for i in range(nh):
            for j in range(no):
                self.wei_ho[i][j]=rand_bet(-0.2,0.2)

        self.bias_h=0.2*ones(nh)
        self.bias_o=0.2*ones(no)
        # 存储上一次的权重变化
        self.wei_chih=zeros([ni,nh])
        self.wei_chho=zeros([nh,no])

    def update(self,input):
        if len(input)!=self.ni:
            print('Input error!')
            return

        for i in range(self.ni):
            self.ai[i]=input[i]

        for i in range(self.nh):
            get_ah=0.0
            for j in range(self.ni):
                get_ah += self.wei_ih[j][i]*self.ai[j]
            self.ah[i] = sigmond(get_ah-self.bias_h[i])

        for i in range(self.no):
            get_ao=0.0
            for j in range(self.nh):
                get_ao+=self.wei_ho[j][i]*self.ah[j]
            self.ao[i] = sigmond(get_ao-self.bias_o[i])
        return self.ao

    def BP(self,y,lamda=1):
        if len(y)!=self.no:
            print('target wrong!')
        g_j=zeros(self.no)
        e_h=zeros(self.nh)
        for i in range(self.no):
            g_j[i] = self.ao[i]*(1.0-self.ao[i])*(y[i]-self.ao[i])
            self.bias_o[i]-=lamda*g_j[i]
        for i in range(self.nh):
            for j in range(self.no):
                self.wei_ho[i][j]+=lamda*g_j[j]*self.ah[i]
                self.wei_chho[i][j]=lamda*g_j[j]*self.ah[i]

        for i in range(self.nh):
            ans_e_h=0.0
            for j in range(self.no):
                ans_e_h+=self.wei_ho[i][j]*g_j[j]
            e_h[i]=self.ah[i]*(1-self.ah[i])*ans_e_h
            self.bias_h-=lamda*e_h[i]
        for i in range(self.ni):
            for j in range(self.nh):
                self.wei_ih[i][j]+=lamda*e_h[j]*self.ai[i]
                self.wei_chih[i][j]=lamda*e_h[j]*self.ai[i]
        return

    def train(self,pattern,iter=200000):
        for i in range(iter):
            for p in pattern:
                self.update(p[0])
                self.BP(p[1])
        return

    def test(self,pattern):
        for p in pattern:
            print(self.update(p[0]))
        return


pat=[[[0,0],[0]],[[0,1],[1]],[[3,4],[1]],[[5,6],[0]]]
n=BP_net(2,2,1)
n.train(pat)
n.test(pat)
