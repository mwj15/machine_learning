#!usr/bin/python
#coding:utf-8
from time import sleep
#添加json库
import json
#添加urllib2库
import urllib2
from numpy import*
import math
#读取文件内容，一行是一组数据，格式均为ｆｌｏａｔ
def loaddateset(filename):
    numfeature=len(open(filename).read(line).split('\t'))-1
    dataset=[];labelset=[]
    fr=open(filename)
    for line in fr.readlines():
        fea=[]
        curline=line.strip( ).split('\t')  #去除空格(strip),get a list
        for i in range(numfeature):
            fea.append(float(curline[i]))
        dataset.append(fea)
        labelset.append(float(curline[-1]))    #get the last one
    return dataset,labelset

#线性规划的核心，求解公式：ｗ＝(ｘ转置　×　ｘ)的逆　×　ｘ转置　×　ｙ
#为了得到ｂ，ｘ的后面都要有一个１才行
#所得的ｗ是一个列向量
"""
def linepro(xarr,yarr):       #arr is array，ｙ是行向量，需要转化为列的
    x=mat(xarr);y=mat(yarr).T      #from array get matrix. y别忘了转置
    xtx=x.T*x                    
    if linalg.det(xtx)==0.0:           #ｎｕｍｐｙ的线性代数库ｌｉｎａｌｇ，ｄｅｔ为求行列式
        print('the matrix is sinigular, can not do inverse')
        return
    ws=xtx.I*(x.T*y)
    return ws　　　　　　　
    #别忘了.I
"""
#numpy内有函数corrcoef可以检测两个数组的相关性，乘ｗ的与真实值可以比一下，注意数组维度的问题

#局部加权线性回归（ＬＷＬＲ），通过给预测点附近的ｘ更高的权重来增加准确度，使用高斯核函数来分配权重
#k越大，所拟合出来的线就越直，很小时几乎都不成样子了
#w(i,i)=exp(-|x(i)-x|/2/k**2)，ｋ自定，得到一个对角线矩阵，公式变为：w*=(xTWx)-1(xTWy)
def lwlr(x,xarr,yarr,k):
    xmat=mat(xarr);ymat=mat(yarr).T
    num=shape(xarr)[0]         #得到样本的数量,[0]是读取一维的长度，直接用shape可以得到两个数，直接看大小
    wei=mat(eye(num))         #初始化ｗｅｉ，别忘了ｍａｔ（）一下，ｅｙｅ后对角线上都是１
    for i in range(num):
        diff=xmat[i,:]-x
        wei[i,i]=exp(diff.T*diff/(-2.0*k**2))     #ｅｘｐ直接用就行，２后面别忘了.0
    xTx=xmat.T*wei*xmat
    if linalg.det(xTx)==0.0:
        print('the matrix is sinigular, can not do inverse')
        return
    ws=x*(xTx.I*(xmat.T*wei*ymat)) 
    return ws         #直接返回测试点的值

#用此函数对测试集进行测试
def testlwlr(testarr,xarr,yarr,k):
    m=shape(testarr)[0]
    yce=zeros(m)
    for i in range(m):
        yce[i]=lwlr(testarr[i],xarr,yarr,k)
    return yce


#zeros相关讲解：
#一个参数时返回一维，两个参数时矩阵，同样用法还有ｏｎｅｓ,empty(未初始化的，里面数字都是随机的)
#相关函数有zeros_like,ones_like,empty_like，返回一个与输入的同规格的

#岭回归模型，用于所得特征值过多，多于所得的向本数量时，采用这种方法来减少不必要的参数
#方式：在矩阵xTx上加入一个λI从而使得矩阵非奇异，进而能对矩阵xTx+λI求逆，w*=(xTx+λI)-1(xTy)
#此外，增加了相关约束：Σwi2<=λ，即回归系数向量中每个参数的平方和不能大于λ，这就避免了当两个或多个特征相关时，可能出现很大的正系数和很大的负系数。
#λ的值通过预测误差最小化的方式来获得，之后讲数据分为测试集和训练集，进行学习
#岭回归非常适合用来解病态的问题和共线性问题，病态问题往往是对角线上的元素太小了引起的，共线性问题是指其中部分因素有精确的相关性，干扰了模型的预测
#岭回归中的数据需要做一下标准化处理，就是将数据的每一维度的特征减去相应的特征均值，然后除以特征的方差

def ridgeRegres(xMat,yMat,lam=0.2):
    #计算矩阵内积
    xTx=xMat.T*xMat
    #添加惩罚项，使矩阵xTx变换后可逆
    denom=xTx+eye(shape(xMat)[1])*lam
    #判断行列式值是否为0，确定是否可逆
    if linalg.det(denom)==0.0:
        print('This matrix is singular,cannot do inverse')
        return 
    #计算回归系数
    ws=denom.I*(xMat.T*yMat)
    return ws

#特征需要标准化处理，使所有特征具有相同重要性
#求解不同的ｌａｍｄａ下的拟合向量值
def ridgeTest(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    #计算均值
    #mean函数：只有一个矩阵参数，返回一个实数，为所有数的平均数，如果参数为０，压缩成１行，对列求均值，参数为１，压缩为１列，对行求均值
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    #计算各个特征的方差
    #xVar=var(xMat,0)　　　　　　　　　#求各列的方差
    #特征-均值/方差
    xMat=(xMat-xMeans)/xVar
    #在30个不同的lamda下进行测试
    numTestpts=30
    #30次的结果保存在wMat中
    #wMat=zeros((numTestpts,shape(xMat)[1]))　　　　#shape后面接１就是求得每个ｘ有多少特征值了，即列数
    for i in range(numTestpts):
        #计算对应lamda回归系数，lamda以指数形式变换
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

#正则化相关讲解
#主要用在数据特征值多而样本比较少的情况下，这时候非常容易出现过拟合的现象，此时使用
#如果我们的参数值对应的比较小的话，旺旺这个假设的形式就会更加简单，不易过拟合，正则化操作就是减小代价函数里每一个参数的值
#形式就是在代价函数后面加一个Ｌａｍｄａ值，如lamda(a^2+b^2+...)这样，lamda比较大时就会限制a和b都比较小，这样就成功防止过拟合了，比如预测房价，在多项式的高次系数上加这个可以防止过拟合

#数据标准化
#将数据按比例缩放，常用方式就是(x-mean(x))/std(x)

#向前逐步回归
def stagewise(xarr,yarr,eps,num):
    xmat=mat(xarr);ymat=mat(yarr)
    ymean=mean(yarr,0)
    ymat=ymat-ymean
    xmean=mean(xmat,0);xvar=var(xmat,0)
    xmat=(xmat-xmean)/xvar
    m,n=shape(xmat)
    returnmat=zeros((m,n))
    ws=zeros((n,1));wst=ws.copy();wsm=ws.copy()
    error=inf
    for i in range(num):
        for j in range(n):
            for sign in-[-1,1]:
                wst=wsm.copy();wst[j]=wst[j]+eps*sign
                ytest=xmat*wst
                diff=ytest-ymat
                newerror=diff.T*diff
                if newerror<error:
                    error=newerror
                    wsm=wst.copy()
        ws=wsm.copy()
        returnmat[i,:]=ws.T
    return returnmat




