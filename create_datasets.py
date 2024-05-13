# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:34:59 2024

@author: shepe
"""

import pandas as pd
import json
from valentine import valentine_match, valentine_metrics
from valentine.algorithms import Coma, Cupid, DistributionBased
import numpy as np
import os
import pickle
from hxl_tag import HXLTagger
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
import textdistance

smoothie = SmoothingFunction().method4

hxl_tagger = HXLTagger()

#ДАТАСЕТ с ограничениями и описательной статистикой

def get_constraints(df_):
    constr = []
    for i in df_.columns:
        nnans = df_[i].isnull().sum()/len(df_[i])
        column = df_[i].dropna()
        type_ = column.dtype
        nunique = column.nunique()
        if type_=='object':
            column = column.fillna('')
            column = column.apply(lambda x: str(x))
            max_len = column.apply(lambda x: len(x)).max()
            min_len = column.apply(lambda x: len(x)).min()
            average_len = column.apply(lambda x: len(x)).mean()
            describe = [0,0,0,0,0,0,0,0]
            t = 1
        else:
            max_len = 0
            min_len = 0
            average_len = 0
            describe = list(column.describe().values)
            t = 0
        
        constr.append([nnans,t,nunique,max_len,min_len,average_len]+describe)
    return constr

def get_data(df1,df2,ground_t1, data_name1='',data_name2=''):
    constr1 = get_constraints(df1)
    constr2 = get_constraints(df2)
    data = []
    for attr1_ind in range(len(df1.columns)):
        attr1 = df1.columns[attr1_ind]
        constr1_ = constr1[attr1_ind]
        match = (attr1,'')
        for ground in ground_t1:
            if attr1 == ground[0]:
                match = ground
        for attr2_ind in range(len(df2.columns)):
            attr2 = df2.columns[attr2_ind]
            constr2_ = constr2[attr2_ind]
            if attr2==match[1]:
                res=1
            else:
                res=0
            data.append([data_name1, data_name2, attr1,attr2,res,constr1_+constr2_])
    return data

def get_data_and_ground(all_data_name1, path_ = r'C:\Users\shepe\Downloads\Valentine-datasets\prospect\Unionable'):
    data_name1 = all_data_name1+'_source'
    data_name2 = all_data_name1+'_target'
    df1 = pd.read_csv(r'{}\{}\{}'.format(path_, all_data_name1,data_name1+'.csv'))
    df2 = pd.read_csv(r'{}\{}\{}'.format(path_, all_data_name1,data_name2+'.csv'))

    with open(r'{}\{}\{}_mapping.json'.format(path_, all_data_name1,all_data_name1), 'r') as f:
        mapping = json.load(f)
    ground_t1 = []
    for i in mapping['matches']:
        ground_t1.append((i['source_column'],i['target_column']))
    return [data_name1,data_name2,df1,df2,ground_t1]

def create_train_data(names, number=''):
    data=[]
    for name in names:
        df_and_ground = get_data_and_ground(name)

        data_name1= df_and_ground[0]
        data_name2= df_and_ground[1]
        df1 = df_and_ground[2]
        df2 = df_and_ground[3]
        ground_t1 = df_and_ground[4]

        data_train = get_data(df1,df2,ground_t1, data_name1=data_name1,data_name2=data_name2)
        data+=data_train
    df_data= pd.DataFrame(data,columns = ['dataset1_name','dataset2_name', 'attr1_name', 'attr2_name', 'attribute_match', 'constraints'] )
    df_data.to_pickle('./datasets/omap/train_tpc{}.p'.format(number))

def create_1():
    names = []
    files = os.listdir(r'C:\Users\shepe\Downloads\Valentine-datasets\prospect\Unionable')
    new_files = files
    for i in range(int(len(files)/2)):
        n = len(new_files)
        idx = np.random.choice(list(range(n)), p=np.ones(n)/n)
        print(idx)
        file = new_files[idx]
        names.append(file)
        if i>0:
            create_train_data(names,i)
        new_files = new_files[:idx]+new_files[idx+1:]
    
    with open('names.p', 'wb') as file:
        pickle.dump(names, file)

#ДАТАСЕТ с instance-based+schema-based+тэги

def get_features(df_):
    features = []
    for i in df_.columns:
        nnans = df_[i].isnull().sum()/len(df_[i])
        column = df_[i].dropna()
        type_ = column.dtype
        nunique = column.nunique()
        if type_=='object':
            column = column.fillna('')
            column = column.apply(lambda x: str(x))
            max_len = column.apply(lambda x: len(x)).max()
            min_len = column.apply(lambda x: len(x)).min()
            average_len = column.apply(lambda x: len(x)).mean()
            describe = [0,0,0,0,0,0,0,0]
            t = 1
        else:
            max_len = 0
            min_len = 0
            average_len = 0
            describe = list(column.describe().values)
            t = 0
        
        features.append([nnans,t,nunique,max_len,min_len,average_len]+describe)
    return features

def jaccard_score(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    if len(set1.union(set2)) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def get_hxl_tags_features(hxl_tags1, hxl_tags2):
    """
    Use rules to calculate features.
    """
    tagname_bleu_score = bleu([hxl_tags1["tagname"]], hxl_tags2["tagname"], smoothing_function=smoothie)
    attribute_jaccard_score = jaccard_score(hxl_tags1["attributes"], hxl_tags2["attributes"])
    hxltags_features = [tagname_bleu_score, attribute_jaccard_score]
    return hxltags_features

def get_features_2(column1, column2):
    column1 = column1.dropna()
    column2 = column2.dropna()
    type_1 = column1.dtype
    type_2 = column2.dtype
    column1 = column1.tolist()
    column2 = column2.tolist()
    
    features = [0]
    
    if type_1=='object' and type_2=='object':
        
        #Jaccard 
        sim_1 = jaccard_score(column1,column2)
        #Levenshtein
        sim_2 = textdistance.levenshtein.normalized_similarity(column1,column2)
        #Hamming
        sim_3 = textdistance.hamming.normalized_similarity(column1,column2)
        #Jaro-Winkler
        sim_4 = textdistance.jaro_winkler.normalized_similarity(column1,column2)
        #Cosine similarity
        sim_5 = textdistance.cosine.normalized_similarity(column1,column2)
        features = [sim_1,sim_2,sim_3,sim_4,sim_5,0]
        
    elif type_1!='object' and type_2!='object':
        corr = np.corrcoef(column1,column2)[0][1]
        features = [0,0,0,0,0,corr]
    else:
        pass
    
    return features


def get_data_f(df1,df2,ground_t1, hxl_tags, data_name1='',data_name2=''):
    f1 = get_features(df1)
    f2 = get_features(df2)
    table1_hxl_tags = hxl_tags[0]
    table2_hxl_tags = hxl_tags[1]
    
    data = []
    for attr1_ind in range(len(df1.columns)):
        attr1 = df1.columns[attr1_ind]
        tag1 = table1_hxl_tags[attr1_ind]
        f1_ = f1[attr1_ind]
        match = (attr1,'')
        for ground in ground_t1:
            if attr1 == ground[0]:
                match = ground
        for attr2_ind in range(len(df2.columns)):
            attr2 = df2.columns[attr2_ind]
            tag2 = table2_hxl_tags[attr2_ind]
            f2_ = f2[attr2_ind]
            if attr2==match[1]:
                res=1
            else:
                res=0
            
            hxl_tags_features = get_hxl_tags_features(tag1, tag2)
            
            #features_2 = get_features_2(df1[attr1], df2[attr2]) 
            
            data.append([data_name1, data_name2, attr1+tag1['full'],attr2+tag2['full'],res,f1_+f2_+hxl_tags_features])
    return data


def create_train_data_f(names,hxl_tags, number=''):
    
    data=[]
    i=0
    for name in names:
        print(i, name)
        df_and_ground = get_data_and_ground(name)

        data_name1= df_and_ground[0]
        data_name2= df_and_ground[1]
        df1 = df_and_ground[2]
        df2 = df_and_ground[3]
        ground_t1 = df_and_ground[4]

        data_train = get_data_f(df1,df2,ground_t1, hxl_tags[i], data_name1=data_name1,data_name2=data_name2)
        data+=data_train
        i+=1
    df_data= pd.DataFrame(data,columns = ['dataset1_name','dataset2_name', 'attr1_name', 'attr2_name', 'attribute_match', 'features'] )
    df_data.to_pickle('./datasets/with_feature_extraction/train_tpc{}.p'.format(number))
    return df_data

def create_2():
    with open('names.p', 'rb') as file:
        names = pickle.load(file)

    hxl_tags=[]
    for j in range(len(names)):
        dfs = get_data_and_ground(names[j])
        df1 = dfs[2]
        df2 = dfs[3]
        t1 = hxl_tagger.get_hxl_tags(df1)
        t2 = hxl_tagger.get_hxl_tags(df2)
        hxl_tags.append([t1,t2])
    for i in range(1,17):
        create_train_data_f(names[:i+1],hxl_tags,i)
        
#ДАТАСЕТ с 1 ограничением

def get_constraints_2(df_):
    constr = []
    for i in df_.columns:
        column = df_[i].dropna()
        type_ = column.dtype
        if type_=='object':
            t = 1
        else:
            t = 0
        
        constr.append([t])
    return constr

def get_data_2(df1,df2,ground_t1, data_name1='',data_name2=''):
    constr1 = get_constraints_2(df1)
    constr2 = get_constraints_2(df2)
    data = []
    for attr1_ind in range(len(df1.columns)):
        attr1 = df1.columns[attr1_ind]
        constr1_ = constr1[attr1_ind]
        match = (attr1,'')
        for ground in ground_t1:
            if attr1 == ground[0]:
                match = ground
        for attr2_ind in range(len(df2.columns)):
            attr2 = df2.columns[attr2_ind]
            constr2_ = constr2[attr2_ind]
            if attr2==match[1]:
                res=1
            else:
                res=0
            data.append([data_name1, data_name2, attr1,attr2,res,constr1_+constr2_])
    return data


def create_train_data_2(names, number=''):
    data=[]
    for name in names:
        df_and_ground = get_data_and_ground(name)

        data_name1= df_and_ground[0]
        data_name2= df_and_ground[1]
        df1 = df_and_ground[2]
        df2 = df_and_ground[3]
        ground_t1 = df_and_ground[4]

        data_train = get_data_2(df1,df2,ground_t1, data_name1=data_name1,data_name2=data_name2)
        data+=data_train
    df_data= pd.DataFrame(data,columns = ['dataset1_name','dataset2_name', 'attr1_name', 'attr2_name', 'attribute_match', 'constraints'] )
    df_data.to_pickle('./datasets/with_1_constraint/train_tpc{}.p'.format(number))
    
def create_3():
    with open('names.p', 'rb') as file:
        names = pickle.load(file)
    for i in range(1,17):
        create_train_data_2(names[:i+1],i)


if __name__ == '__main__':
    create_1()
    create_2()
    create_3()