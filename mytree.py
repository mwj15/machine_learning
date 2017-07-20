#!usr/bin/python
#coding:utf-8
from math import log
class tr:
# claculate the entropy of a dataset
# the dataset is a list, the member of it is also list
    def __init__(self,dataset,labels):
        self.tree=self.createtree(dataset,labels)

    def cal_entropy(self,dataset):
        num=len(dataset)
        label_c={}
        for fea in dataset:
            cur=fea[-1]
            #print(cur)
            if cur not in label_c.keys():
                label_c[cur]=0
            label_c[cur]+=1
        entropy=0.0
        for key in label_c:
            cur_cal=float(label_c[key])/num
            entropy-=cur_cal*log(cur_cal,2)
        return entropy

    # get a new dataset from the old one according to the feature
    #axis is the feature you choose and the value is what the feature in this dataset is
    def get_newdata(self,dataset,axis,value):
        newdata=[]
        for fea in dataset:
            if fea[axis]==value:
                # than delete the feature and throw it into the new dataset
                new_fea=fea[:axis]
                new_fea.extend(fea[axis+1:])
                #print('newfea',new_fea)
                newdata.append(new_fea)
                #print(newdata)
        return newdata

    # choose spilt the dataset according to which feature
    def get_feature(self,dataset):
        numfea=len(dataset[0])-1      # get the number of the feature left
        base_entropy=self.cal_entropy(dataset)
        entropy_gain=0.0
        best_fea=-1
        for i in range(numfea):
            new_gain=0
            val_fea=[example[i]for example in dataset]
            uni_fea=set(val_fea)
            for it in uni_fea:
                new_data=self.get_newdata(dataset,i,it)
                prob=len(new_data)/float(len(dataset))
                new_data_entropy=prob * self.cal_entropy(new_data)
                new_gain+=new_data_entropy
            #　信息增益是旧的减去新的总和，越大越好，这里反过来了
            if new_gain-base_entropy < entropy_gain:
                entropy_gain=new_gain
                best_fea=i
        return best_fea

    #　当所有的特征都用完了时，如果还有结果不一样的，这时候的最终值取决于较多的那一种
    def class_vote(self,class_list):
        class_count={}
        for item in class_list:
            if item not in class_count.keys():
                class_count[item]=0
            class_count[item]+=1
        new_class_count=sorted(class_count.iteritems(),key=operator.itemgetter(1),reverse=True)#可以查一下，第二个参数代表排序的依据
        return new_class_count[0][0]

    # create a new tree
    def createtree(self,dataset,labels):
        class_list=[example[-1] for example in dataset]
        if class_list.count(class_list[0])==len(class_list):
            return class_list[0]                        #　所有类都一样了
        if len(dataset[0])==1:#　所有特点都用光了
            return self.class_vote(class_list)
        best_fea=self.get_feature(dataset)
        best_fea_label=labels[best_fea]
        mytree={best_fea_label:{}}
        sub_label = labels[:]
        del(sub_label[best_fea])
        fea_val=[example[best_fea]for example in dataset]
        fea_val_uni=set(fea_val)
        for value in fea_val_uni:
            mytree[best_fea_label][value]=self.createtree(self.get_newdata(dataset,best_fea,value),sub_label)
        #print(mytree)
        return mytree

    # classify according to the tree
    def classify(self,labels,fea_vec):
        fir_fea=self.tree.keys()[0]
        print(fir_fea)
        second_dict=self.tree[fir_fea]
        print(labels)
        fir_index=labels.index(fir_fea)
        for key in second_dict:
            if fea_vec[fir_index]==key:
                if type(second_dict[key]).__name__=='dict':
                    return self.classify(second_dict,labels,fea_vec)
                else:
                    return second_dict[key]

dataset = [[1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
labels = ['a', 'b']
mytree=tr(dataset,labels)
print(mytree.tree)
class_=mytree.classify(labels,[1,0])
print(class_)