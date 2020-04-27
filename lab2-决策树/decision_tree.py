import numpy as np
import csv
import matplotlib.pyplot as plt

def get_empirical_entropy(data_set, label_col):
    #获取数据集data_set的经验熵
    label_dict={}
    data_num=len(data_set)
    for row in data_set:
        if row[label_col] not in label_dict:
            label_dict[row[label_col]]=1
        else:
            label_dict[row[label_col]]+=1
    empirical_entropy=0
    for value in label_dict.values():
        empirical_entropy-=value/data_num*np.log(value/data_num)
    return empirical_entropy

def get_conditional_entropy(data_set, label_col, A):
    #计算特征A对数据集data_set的条件熵
    A_kind_dict={}
    for row in data_set:
        if row[A] not in A_kind_dict:
            A_kind_dict[row[A]]=[row]
        else:
            A_kind_dict[row[A]].append(row)
    conditional_entropy=0
    for key in A_kind_dict.keys():
        A_kind_dict[key]=np.array(A_kind_dict[key])
        conditional_entropy+=len(A_kind_dict[key])/len(data_set)*get_empirical_entropy(A_kind_dict[key], label_col)
    return conditional_entropy

def get_information_gain(data_set, label_col, A):
    #信息增益=经验熵-条件熵
    return get_empirical_entropy(data_set, label_col)-get_conditional_entropy(data_set, label_col, A)

def get_information_gain_ratio(data_set, label_col, A):
    #计算信息增益率
    return get_information_gain(data_set, label_col, A)/get_empirical_entropy(data_set, label_col)

def get_gini_index(data_set, label_col, A):
    #计算gini系数(二分类)
    #函数返回计算所得的gini系数，和该属性中能使gini系数最小的二分类属性取值
    A_kind_dict={}
    A_kind_dict_one={}
    for row in data_set:
        if row[A] not in A_kind_dict.keys():
            A_kind_dict[row[A]]=1
            if row[label_col]=='1':
                A_kind_dict_one[row[A]]=1
            else:
                A_kind_dict_one[row[A]]=0
        else:
            A_kind_dict[row[A]]+=1
            if row[label_col]=='1':
                A_kind_dict_one[row[A]]+=1
    gini_index=0
    for kind in A_kind_dict.keys():
        one_pro=A_kind_dict_one[kind]/A_kind_dict[kind]
        gini_index+=A_kind_dict[kind]/len(data_set)*(1-one_pro**2-(1-one_pro)**2)
    return gini_index
    
#    for key in A_kind_dict.keys():
#        total_one=0
#        for other_key in A_kind_dict.keys():
#            if other_key!=key:
#                total_one+=A_kind_dict_one[other_key]
#        key_one_pro=A_kind_dict_one[key]/A_kind_dict[key]
#        other_key_one_pro=total_one/(len(data_set)-A_kind_dict[key])
#        gini_index=A_kind_dict[key]/len(data_set)*(1-key_one_pro**2-(1-key_one_pro)**2)
#        gini_index+=(1-A_kind_dict[key]/len(data_set))*(1-other_key_one_pro**2-(1-other_key_one_pro)**2)
#        if gini_index<min_gini_index:
#            min_gini_index=gini_index
#            min_kind=key
#    return (min_gini_index, min_kind)


def Select_Attribution(data_set, mood, attribution_list):
    #按照给定的mood对划分属性进行选择
    #找到最优的划分属性并返回
    label_col=len(data_set[0])-1
    if mood=='ID3':
        max_information_gain=float('-inf')
        max_attribution=None
        for attribution in attribution_list:
            if get_information_gain(data_set, label_col, attribution)>max_information_gain:
                max_information_gain=get_information_gain(data_set, label_col, attribution)
                max_attribution=attribution
        return max_attribution
    elif mood=='C4.5':
        max_information_gain_ratio=float('-inf')
        max_attribution=None
        for attribution in attribution_list:
            if get_information_gain_ratio(data_set, label_col, attribution)>max_information_gain_ratio:
                max_information_gain_ratio=get_information_gain_ratio(data_set, label_col, attribution)
                max_attribution=attribution
        return max_attribution
    elif mood=='CART':
        min_gini_index=float('inf')
        min_attribution=None
        for attribution in attribution_list:
            if get_gini_index(data_set, label_col, attribution)<min_gini_index:
                min_gini_index=get_gini_index(data_set, label_col, attribution)
                min_attribution=attribution
        return min_attribution               
#        min_kind=None
#        for attribution in attribution_list:
#            if get_gini_index(data_set, label_col, attribution)[0]<min_gini_index:
#                (min_gini_index, min_kind)=get_gini_index(data_set, label_col, attribution)
#                min_attribution=attribution
#        return (min_attribution, min_kind)

def Data_Set_Partition(data_set, label, A):
    #对数据集根据特征A进行划分，返回划分之后的列表
    partition_dict={}
    for row in data_set:
        if row[A] not in partition_dict.keys():
            partition_dict[row[A]]=[]
    for row in data_set:
        partition_dict[row[A]].append(row)
    for key in partition_dict.keys():
        partition_dict[key]=np.array(partition_dict[key])
    return partition_dict

attribution_dict={ 0 : ['vhigh', 'high', 'med', 'low'],      #buying
                   1 : ['vhigh', 'high', 'med', 'low'],      #maint
                   2 : ['2', '3', '4', '5more'],             #dorrs
                   3 : ['2', '4', 'more'],                   #persons
                   4 : ['big', 'med', 'small'],              #lug_boot
                   5 : ['high', 'med', 'low']                #safety
                 }



class Node:
    def __init__(self, split_attribution=None, attribution_kind=None, label=None):
        #split_attribution用于标记该Node的分裂属性
        #attribution_kind用于标记父节点分裂后该结点对应分裂属性的取值
        #label用于标记Node的预测分类，非叶子节点则为None
        #child_nodes用于存放Node的子节点
        #is_leaf用于标记该结点是否为叶子节点
        self.split_attribution=split_attribution
        self.attribution_kind=attribution_kind
        self.label=label
        self.child_nodes=[]
        self.is_leaf=False

class DecisionTree:
    def __init__(self, mood):
        self.mood=mood

    def is_same_in_label(self, data_set):
        #判断数据集是否属于同一个label
        label_col=len(data_set[0])-1
        for row in data_set:
            if row[label_col]!=data_set[0][label_col]:
                return False
        return True

    def is_same_in_all_attributions(self, data_set, attribution_list):
        #判断数据集是否在所有属性取值上相同
        label_col=len(data_set[0])-1
        data_set=data_set[:,:label_col]
        for row in data_set:
            for col in attribution_list:
                if data_set[row][col]!=data_set[0][col]:
                    return False
        return True

    def get_max_label(self, data_set):
        #获取data_set中最多的label
        label_col=len(data_set[0])-1
        label_dict={'0':0, '1':0}
        for row in data_set:
            label_dict[row[label_col]]+=1
        if label_dict['0']>label_dict['1']:
            return '0'
        else:
            return '1'

    def build_tree(self, data_set, root, attribution_list):
        #如果data_set都属于一个标签，则将其标为该标签的叶节点
        label_col=len(data_set[0])-1
        if self.is_same_in_label(data_set):
            root.label=data_set[0][label_col]
            root.is_leaf=True
            return

        #如果属性集为空或者在所有属性上取值相同，将当前节点标记为叶节点，类别为D出现最多的类
        if len(attribution_list)==0 or self.is_same_in_label(data_set):
            root.label=self.get_max_label(data_set)
            root.is_leaf=True
            return


        attribution=Select_Attribution(data_set, self.mood, attribution_list)
        root.split_attribution=attribution
        for kind in attribution_dict[attribution]:
            sub_data_set=[]
            for row in data_set:
                if row[attribution]==kind:
                    sub_data_set.append(row)
            sub_data_set=np.array(sub_data_set)
            #如果数据集为空，则将当前节点标记为叶节点，类别为父类中出现最多的类
            if len(sub_data_set)==0:
                sub_tree=Node(-1, kind, self.get_max_label(data_set))
                sub_tree.is_leaf=True
                root.child_nodes.append(sub_tree)
            else:
                new_attribution_list=[]
                for item in attribution_list:
                    if item!=attribution:
                        new_attribution_list.append(item)
                sub_tree=Node(-1, kind, self.get_max_label(sub_data_set))
                self.build_tree(sub_data_set, sub_tree, new_attribution_list)
                root.child_nodes.append(sub_tree)




def classification(root, test_sample):
    if root.is_leaf:
        if int(root.label)==int(test_sample[6]):
            return True
        else:
            return False
    split_attribution=root.split_attribution
    for child in root.child_nodes:
        attribution_kind=child.attribution_kind
        if test_sample[split_attribution]==attribution_kind:
            return classification(child, test_sample)


def cal_accuracy(root, test_set):
    count = 0
    for row in test_set:
        if classification(root, row):
            count += 1
    accuracy = count / len(test_set)
    return accuracy

def Five_Cross_Validation(data_set):
    #将数据集打乱
    row_rand_array = np.arange(data_set.shape[0])
    np.random.shuffle(row_rand_array)
    set_size=int(0.2*len(data_set))
    set_list=[]
    set_list.append(data_set[row_rand_array[0:set_size]])
    set_list.append(data_set[row_rand_array[set_size:2*set_size]])
    set_list.append(data_set[row_rand_array[2*set_size:3*set_size]])
    set_list.append(data_set[row_rand_array[3*set_size:4*set_size]])   
    set_list.append(data_set[row_rand_array[4*set_size:]])
    avg_accuracy_ID3=0
    avg_accuracy_C45=0
    avg_accuracy_CART=0
    #五折交叉验证计算平均准确率
    for index in range(len(set_list)):
        test_set=set_list[index]
        train_set_list=[set_list[i] for i in range(len(set_list)) if i!=index]
        train_set=train_set_list[0]
        for i in range(1,len(train_set_list)):
            train_set=np.vstack((train_set,train_set_list[i]))        
        ID3_tree=DecisionTree('ID3')
        C45_tree=DecisionTree('C4.5')
        CART_tree=DecisionTree('CART')
        ID3_root=Node()
        C45_root=Node()
        CART_root=Node()
        ID3_tree.build_tree(train_set, ID3_root, [0,1,2,3,4,5])
        C45_tree.build_tree(train_set, C45_root, [0,1,2,3,4,5])
        CART_tree.build_tree(train_set, CART_root, [0,1,2,3,4,5])
        accuracy_ID3=cal_accuracy(ID3_root, test_set)
        accuracy_C45=cal_accuracy(C45_root, test_set)
        accuracy_CART=cal_accuracy(CART_root, test_set)
        avg_accuracy_ID3+=accuracy_ID3
        avg_accuracy_C45+=accuracy_C45
        avg_accuracy_CART+=accuracy_CART
    avg_accuracy_ID3/=5
    avg_accuracy_C45/=5
    avg_accuracy_CART/=5
    print('ID3:', avg_accuracy_ID3)
    print('C4.5:', avg_accuracy_C45)
    print('CART:', avg_accuracy_CART)
    return (avg_accuracy_ID3, avg_accuracy_C45, avg_accuracy_CART)

def draw_scatter(data_set):
    ID3=[]
    C45=[]
    CART=[]
    for i in range(100):
        (a, b, c)=Five_Cross_Validation(data_set)
        ID3.append(a)
        C45.append(b)
        CART.append(c)
    ID3=np.array(ID3)
    C45=np.array(C45)
    CART=np.array(CART)
    
    print('Average')
    print('ID3:', np.mean(ID3))
    print('C4.5:', np.mean(C45))
    print('CART:', np.mean(CART))
    
    plt.title('Accuracy Scatter Plot of ID3 Modle')
    plt.xlabel('sequence number')
    plt.ylabel('accuracy')
    plt.scatter(range(100), ID3, c="r", marker='x', label='ID3')
    plt.legend()
    plt.show()   
    
    plt.title('Accuracy Scatter Plot of C4.5 Modle')
    plt.xlabel('sequence number')
    plt.ylabel('accuracy')
    plt.scatter(range(100), C45, c="b", marker='+', label='C4.5')
    plt.legend()
    plt.show()   
    
    plt.title('Accuracy Scatter Plot of CART Modle')
    plt.xlabel('sequence number')
    plt.ylabel('accuracy')
    plt.scatter(range(100), CART, c="y", marker='.', label='CART')
    plt.legend()
    plt.show()
        
        
    plt.title('Accuracy Scatter Plot')
    plt.xlabel('sequence number')
    plt.ylabel('accuracy')
    plt.scatter(range(100), ID3, c="r", marker='x', label='ID3')
    plt.scatter(range(100), C45, c="b", marker='+', label='C4.5')
    plt.scatter(range(100), CART, c="y", marker='.', label='CART')
    plt.legend()
    plt.show()       
    
    
if __name__=='__main__':
    file = open('C:/Users/Administrator/Desktop/人工智能实验/Lab2 决策树/lab2_dataset/car_train.csv', 'r')
    file_csv = csv.reader(file)
    file_list = list(file_csv)
    file_list = file_list[1:]
    data_set=np.array(file_list)
    data_num=len(data_set)

#    row_rand_array = np.arange(data_set.shape[0])
#    np.random.shuffle(row_rand_array)
#    test_set = data_set[row_rand_array[0:int(0.2*data_num)]]
#    train_set = data_set[row_rand_array[int(0.2*data_num):data_num]]
    train_set = data_set[:]
    test_set = data_set[:]
    tree=DecisionTree('ID3')
    root=Node()
    tree.build_tree(train_set, root, [0,1,2,3,4,5])
    print('ID3:', cal_accuracy(root, test_set))

#

