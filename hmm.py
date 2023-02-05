from collections import defaultdict
import itertools
from scipy.special import logsumexp
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import math

def load_data(filename):
    ner_tags = []
    f = open(filename)
    
    l = f.readline()
    sen=[]
    while l:
        line = l.strip()
        if line: 
            fields = line.split(" ")
            sen.append(fields)
            #print(sen)
        else: # Empty line
            sen.append(['</s>', '<end>'])
            ner_tags.append(sen)
            sen=[]
        l = f.readline()
        

    #################################################################################
    # TODO: load data in ner_tags
    #################################################################################

    #################################################################################

    return ner_tags


class HMMNER:
    """HMM for NER tagging."""

    def __init__(self):
        self.initial_count = None
        self.emission_count = None
        self.transition_count = None
        self.ner_tags = None
        self.observations = None
        self.tag_to_index = None
        self.observation_to_index = None
        self.initial_prob = None
        self.transition_prob = None
        self.emission_prob = None
        self.word_count={}
        self.tag_count={}

    def get_counts(self, train_data):
        #self.ner_tags = train_data
        self.initial_count={}
        self.emission_count={}
        self.transition_count={}
        #self.initial_count['O'] = 0
        #self.initial_count['B-PER'] = 0
        #self.initial_count['B-ORG'] = 0
        #self.initial_count['B-LOC'] = 0
        #self.initial_count['B-MISC'] = 0
        #self.initial_count['I-PER'] = 0
        #self.initial_count['I-ORG'] = 0
        #self.initial_count['I-LOC'] = 0
        #self.initial_count['I-MISC'] = 0
        #self.initial_count['<end>'] = 0
        self.word_count={}
        self.tag_count={}
        
        for sen in train_data:
            for word in sen:
                w=word[0]
                tag=word[1]
                self.word_count[w]=self.word_count.get(w,0)+1
                self.tag_count[tag]=self.tag_count.get(tag,0)+1
            
        for sen in train_data:
            self.initial_count[sen[0][1]]=self.initial_count.get(sen[0][1],0)+1
            for word in sen:
                tag=word[1]
                w=word[0]
                tu=(tag,w)
                if self.word_count[w]!=1:
                    self.emission_count[tu]=self.emission_count.get(tu,0)+1
                else:
                    tu1=(tag,'UNK')
                    del self.word_count[w]
                    self.word_count['UNK']=self.word_count.get('UNK',0)+1
                    self.emission_count[tu1]=self.emission_count.get(tu1,0)+1
            for i in range(1,len(sen)):
                #print(len(sen))
                tagh=sen[i][1]
                tagq=sen[i-1][1]
                tu=(tagq,tagh)
                self.transition_count[tu]=self.transition_count.get(tu,0)+1
                
        #print(self.initial_count)
            
                

        #################################################################################
        # TODO: store counts to self.initial_count, self.emission_count, self.transition_count
        # initial count
        #################################################################################

        pass

        #################################################################################
    
    def get_lists(self):
        #self.word_count['UNK']=0
        self.ner_tags=[]
        self.tag_to_index={}
        self.observations=[]
        self.observation_to_index={}
        tags=list(self.tag_count.keys())
        obs=list(self.word_count.keys())
        tags.sort()
        obs.sort()
        self.ner_tags=tags
        self.observations=obs
        for i in range(len(self.ner_tags)):
            self.tag_to_index[self.ner_tags[i]]=i
        for i in range(len(self.observations)):
            self.observation_to_index[self.observations[i]]=i

        #################################################################################
        # TODO: store ner tags and vocabulary to self, store their maps to index
        #################################################################################

        pass

        #################################################################################
    
    def get_probabilities(self, initial_k, transition_k, emission_k):
        self.initial_prob = np.zeros(len(self.ner_tags))
    
        self.transition_prob = np.zeros((len(self.ner_tags),len(self.ner_tags)))
        self.emission_prob = np.zeros((len(self.ner_tags),len(self.observations)))
        
        
        for tag in self.ner_tags:
            self.initial_count[tag]=self.initial_count.get(tag,0)+initial_k
        sum_tag=sum([i for i in self.initial_count.values()])
        for tag in self.ner_tags:
            self.initial_prob[self.tag_to_index[tag]]=self.initial_count.get(tag,0)/sum_tag
            self.initial_count[tag]=self.initial_count.get(tag,0)-initial_k
        #print(self.initial_prob)
        #norm
        
        #?????? what the fuxk
        
        #sum2=sum([i for i in self.transition_count.values()])
        #print(sum2)
        # it is 16 correct
        tag_sum=np.zeros(len(self.ner_tags))
        #print(tag_sum)
        #print(self.transition_count)
        for tagq in self.ner_tags:
            for tagh in self.ner_tags:
                self.transition_count[(tagq,tagh)]=self.transition_count.get((tagq,tagh),0)+transition_k
                tag_sum[self.tag_to_index[tagq]]=self.transition_count[(tagq,tagh)]+tag_sum[self.tag_to_index[tagq]]
                
        #print(tag_sum)
        #print(self.transition_count)

        
        for tagq in self.ner_tags:
            for tagh in self.ner_tags:
                self.transition_prob[self.tag_to_index[tagq],self.tag_to_index[tagh]]=(self.transition_count[(tagq,tagh)]/tag_sum[self.tag_to_index[tagq]])
                self.transition_count[(tagq,tagh)]=self.transition_count.get((tagq,tagh),0)-transition_k
        
            
        tag_sum=np.zeros(len(self.ner_tags))  
        for tag in self.ner_tags:
            for word in self.observations:
                self.emission_count[(tag,word)]=self.emission_count.get((tag,word),0)+emission_k
                tag_sum[self.tag_to_index[tag]]=self.emission_count[(tag,word)]+tag_sum[self.tag_to_index[tag]]
        
        
        for tag in self.ner_tags:
            for word in self.observations:
                self.emission_prob[self.tag_to_index[tag],self.observation_to_index[word]]=(self.emission_count[(tag,word)]/tag_sum[self.tag_to_index[tag]])
                self.emission_count[(tag,word)]=self.emission_count.get((tag,word),0)-emission_k
        
        
            
        
        
        #################################################################################
        # TODO: store probabilities in self.initial_prob, self.transition_prob, 
        # and self.emission_prob
        #################################################################################
        
        pass

        #################################################################################

    def beam_search(self, observations, beam_width, should_print=False):
        
        #################################################################################
        # TODO: predict NER tags, you can assume observations are already tokenized
        #################################################################################
        for i in range(len(observations)):
            if self.word_count.get(observations[i],0)==0:
                observations[i]='UNK'
                
        prob = np.zeros((len(self.ner_tags),len(observations)))
        tags =  np.zeros((beam_width,len(observations)),dtype=int)
        backtrace = np.zeros((beam_width,len(observations)),dtype=int)
        
        # first column：
        for i in range(len(self.ner_tags)):
            prob[i,0]=math.log(self.initial_prob[i])+math.log(self.emission_prob[i,self.observation_to_index[observations[0]]])
        col=prob[:,0]
        #print(col)
        index=[i[0] for i in sorted(enumerate(col), key=lambda x:x[1],reverse=True)]
        #print(index)
        for i in range(beam_width):
            tags[i][0]=index[i]
        #print(tags)
        # for later colnmns:
        for i in range(1,len(observations)):
            cur_backtrace=[]# 这里每一个tag j都对应一个从哪来的
            for j in range(len(self.ner_tags)):
                cur_max=[]
                for k in range(beam_width):
                    #i 是列数， j是行数，也就是现在在考虑的tag,后面的tag, k是照tags[i-1]里面的 tag的index, 全部跑一遍的最大
                    
                    #print(prob)
                    x=prob[tags[k,i-1],i-1]
                    x=x+math.log(self.transition_prob[tags[k,i-1],j])
                    x=x+math.log(self.emission_prob[j,self.observation_to_index[observations[i]]])
                    cur_max.append(x)
                max_value = max(cur_max)
                #find index of max value in list 
                max_index = cur_max.index(max_value)
                cur_backtrace.append(max_index)
                prob[j,i]=max_value
            col=prob[:,i]
            index=[i[0] for i in sorted(enumerate(col), key=lambda x:x[1],reverse=True)]
            for hang in range(beam_width):
                tags[hang][i]=index[hang]
                backtrace[hang][i]=cur_backtrace[index[hang]]
        ner_tag=[]
        ner_tag.append(self.ner_tags[tags[0,len(observations)-1]])
        
        
       
        
        index=len(observations)-1
        cur=backtrace[0][index]
        while index>=1:
            index=index-1
            ner_tag.append(self.ner_tags[tags[cur,index]])
            cur=backtrace[cur][index]
        
                
        #################################################################################

        if should_print:
            print('Tag Index Matrix:\n', tags.astype(int))
            print('Backtrace Matrix:\n', backtrace.astype(int))
        ner_tag.reverse()
        return ner_tag

    
    def predict_ner_all(self, sentences, beam_width):
        # sentences is a list of sentences (each sentence is a list of tokens)
        results = []
        for sen in sentences:
            ner_tags = self.beam_search(sen, beam_width, should_print=False)
            results.append(ner_tags)

        #################################################################################
        # TODO: append ner tags for each sentence to results
        #################################################################################

        #################################################################################
        
        return results
    
    def search_k(self, val, beam_width):
        sentences=[]
        labels=[]
        for l in val:
            sen=[]
            label=[]
            for e in l:
                sen.append(e[0])
                label.append(e[1])
            sentences.append(sen)
            labels.append(label)
        #print(type(sentences))
        #print(type(labels))
            
            
            
        initial_k, transition_k, emission_k = 0, 0, 0
        best_acc = 0
        for i in np.arange(0.8,1.4,0.2):
            for j in np.arange(0.8,1.4,0.2):
                for k in np.arange(0.4,0.6,0.1):
                    #if best_acc>0.9:
                        #break
                    #print(i,j,k)
                    self.get_probabilities(i, j, k)
                    predictions=self.predict_ner_all(sentences,beam_width)
                    #print(type(predictions))
                    cur=get_accuracy(predictions,labels)
                    
                    #print(cur)
                    if cur > best_acc:
                        best_acc=cur
                        initial_k=i
                        transition_k=j
                        emission_k=k
                        #print(cur)
                        
                    
                    

        #################################################################################
        # TODO: search for the best combination of k values
        #################################################################################

        #################################################################################
        
        print(f"Best accuracy: {best_acc}")

        return initial_k, transition_k, emission_k

    def search_beam_width(self, initial_k, transition_k, emission_k, beam_widths, val):
        best_beam_width = -1
        accuracies = []
        self.get_probabilities(initial_k, transition_k, emission_k)
        sentences=[]
        labels=[]
        best=0
        for l in val:
            sen=[]
            label=[]
            for e in l:
                sen.append(e[0])
                label.append(e[1])
            sentences.append(sen)
            labels.append(label)
            
        for width in beam_widths:
            predictions=self.predict_ner_all(sentences,width)
            cur=get_accuracy(predictions,labels)
            accuracies.append(cur) 
            if cur>best:
                best=cur
                best_beam_width=width
            
              
        #################################################################################
        # TODO: search for the best beam width
        #################################################################################

        #################################################################################

        for i in range(len(beam_widths)):
            print(f"Beamwidth = {beam_widths[i]}; Accuracy = {accuracies[i]}")

        return best_beam_width

    def test(self, initial_k, transition_k, emission_k, beam_width, test):
        accuracy = 0
        sentences=[]
        labels=[]
        
        for l in test:
            sen=[]
            label=[]
            for e in l:
                sen.append(e[0])
                label.append(e[1])
            sentences.append(sen)
            labels.append(label)
        self.get_probabilities(initial_k, transition_k, emission_k)
        predictions=self.predict_ner_all(sentences, beam_width)
        accuracy=get_accuracy(predictions,labels)
        #################################################################################
        # TODO: get accuracy on the test set
        #################################################################################

        #################################################################################

        return accuracy

    def forward_algorithm(self, observations):
        prob = 0
        
        for i in range(len(observations)):
            if self.word_count.get(observations[i],0)==0:
                observations[i]='UNK'
                
        probs = np.zeros((len(self.ner_tags),len(observations)))
        
        for i in range(len(self.ner_tags)):
            probs[i,0]=math.log(self.initial_prob[i]*self.emission_prob[i,self.observation_to_index[observations[0]]])
        
        for col in range(1,len(observations)):
            # col represent the word
            word= observations[col]
            c1=probs[:,col-1]
            
            for row in range(len(self.ner_tags)):
                # row represent the current tag
                c2=self.transition_prob[:,row]
                #print(c2)
                
                c3=[self.emission_prob[row,self.observation_to_index[word]]]*len(self.ner_tags)
                
                
                c=[math.exp(c1[k])*c2[k]*c3[k] for k in range(len(self.ner_tags))]
                s=sum(c)
                p=math.log(s)
                probs[row,col]=p
                
                
        last=probs[:,len(observations)-1]
        prob=logsumexp(last)
                
                
                

        #################################################################################
        # TODO: return the probability of sentence given the HMM you have created
        #################################################################################

        

        #################################################################################

        print('Log Probability Matrix:\n', probs.astype(int))

        return prob

def get_accuracy(predictions, labels):
    accuracy = 0
    total=0
    correct=0
    
    for i in range(len(labels)):
        total+=len(labels[i])
        for j in range(len(labels[i])):
            if labels[i][j]==predictions[i][j]:
                correct+=1
    
    accuracy=correct/total

    #################################################################################
    # TODO: calculate accuracy
    #################################################################################

    #################################################################################

    return accuracy