#!usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lean_Guan

import re
import time
import jieba
import logging
import threading
import configparser
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


class Emotion_analysis(object):
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs   # 线程数
        self.conf_path = './aaa.ini'
        self.emotion_path = self.read_conf().get('configure_file', 'emotion_path')
        self.conf = self.read_conf()
        self.host = self.conf.get('configure_db', 'host')
        self.user = self.conf.get('configure_db', 'user')
        self.password = self.conf.get('configure_db', 'password')
        self.port = self.conf.get('configure_db', 'port')
        self.db = self.conf.get('configure_db', 'db')
        self.database = self.conf.get('configure_content', 'read_type')

    def read_conf(self):
        conf = configparser.ConfigParser()
        file = open(self.conf_path, encoding='utf8')
        conf.read_file(file)
        return conf

    def emotion_dict(self, content):
        # path = r"D:/Download/jiangsu/"
        path = self.emotion_path
        word_dict = {}
        content_tmp = content.replace('\r\n', '').replace(' ', '').replace('\n', '')
        content_seg = jieba.cut(content_tmp.strip())
        stopwords = [line.strip() for line in open(path + 'stopwords.txt', 'r', encoding='utf-8').readlines()]
        result1 = []
        result = []
        pattern = re.compile(r'[^A-Za-z0-9_\n]*', re.I)
        for word in content_seg:
            if word not in stopwords:
                word = re.search(pattern, word)
                outstr = ''
                if word != '\t' and word:
                    word = word.group()
                    outstr += word
                    outstr += ''
                result1.append(outstr)
                for i in range(0, len(result1)):
                    word_dict[result1[i]] = i
        for j in word_dict.keys():
            result.append(j)
        # print('===========原始==========')
        # print(word_dict)
        # print(result)
        return word_dict, result

    def words(self):
        # path = r"D:/Download/jiangsu/raw_data/"
        path = self.emotion_path
        sen_file = open(path + r'BosonNLP_sentiment_score.txt', encoding='utf-8')

        sen_list = sen_file.readlines()
        sen_dict = dict()
        for s in sen_list:
            sen_dict[s.split(' ')[0]] = s.split(' ')[1].strip()

        not_word_file = open(path + r"not.txt", encoding='utf-8')
        not_list = not_word_file.readlines()
        not_list = [i.strip() for i in not_list]
        degree_file = open(path + r"degree.txt", encoding='utf-8')
        degree_list = degree_file.readlines()
        # degree_list = read_file(path + 'degree.txt')
        degree_dict = dict()
        for d in degree_list:
            degree_dict[d.split(',')[0]] = d.split(',')[1].strip()
        return sen_dict, not_list, degree_dict

    # 原始找出情感词、程度副词、否定词
    # def classify_words(self, word_list, sen_dict, not_list, degree_dict):
    #     sen_word = dict()
    #     not_word = dict()
    #     degree_word = dict()
    #     # print(emotion_dict())
    #     for i in range(len(word_list)):
    #         word = word_list[i]
    #         if word in sen_dict.keys() and word not in not_list and word not in degree_dict.keys():
    #             sen_word[i] = sen_dict[word]
    #         elif word in not_list and word not in degree_dict.keys():
    #             not_word[i] = -1
    #         elif word in degree_dict.keys():
    #             degree_word[i] = degree_dict[word]
    #     # print('========情感词===========')
    #     # print(sen_word)
    #     # print(not_word)
    #     # print(degree_word)
    #     return sen_word, not_word, degree_word
    """原始找出情感词、程度副词、否定词"""
    def classify_words(self, word_list, sen_dict, not_list, degree_dict):
        sen_word = dict()
        not_word = dict()
        degree_word = dict()
        # print(emotion_dict())
        for i in word_list:
            # word = word_list[i]
            if i in sen_dict.keys() and i not in not_list and i not in degree_dict.keys():
                sen_word[i] = sen_dict[i]
            elif i in not_list and i not in degree_dict.keys():
                not_word[i] = -1
            elif i in degree_dict.keys():
                degree_word[i] = degree_dict[i]
        # print('========情感词===========')
        # print(sen_word)
        # print(not_word)
        # print(degree_word)
        return sen_word, not_word, degree_word

    # 原始计算得分
    # def score_sent(self, sen_word, not_word, degree_word, seg_result):
    #     w = .1
    #     score = 0
    #     sen_loc = list(sen_word.keys())
    #     # print(sen_loc)
    #     not_loc = not_word.keys()
    #     degree_loc = degree_word.keys()
    #     senloc = -1
    #     # notloc = -1
    #     # degreeloc = -1
    #     for i in range(0, len(seg_result)):
    #         # 如果该词为情感词
    #         if i in sen_loc:
    #             # loc为情感词位置列表的序号
    #             senloc += 1
    #             # 直接添加该情感词分数
    #             score += w * float(sen_word[i])
    #             # print("score = %f" % score)
    #             if senloc < len(sen_loc) - 1:
    #                 # 判断该情感词与下一情感词之间是否有否定词或程度副词
    #                 # j为绝对位置
    #                 for j in range(sen_loc[senloc], sen_loc[senloc + 1]):
    #                     # 如果有否定词
    #                     if j in not_loc:
    #                         w *= -1
    #                     # 如果有程度副词
    #                     elif j in degree_loc:
    #                         w *= int(degree_word[j])
    #         # i定位至下一个情感词
    #         if senloc < len(sen_loc) - 1:
    #             i = sen_loc[senloc + 1]
    #
    #     logging.info((str(id) + '------->  score = {}'.format(score)))
    #
    #     return score

    """新计算得分"""
    # def score_sent(self, sen_word, not_word, degree_word, seg_result):
    #     w = .1
    #     score = 0
    #     print('sen_word:', sen_word, 'not_word:', not_word, 'degree_word:', degree_word)
    #
    #     sen_loc = list(sen_word.keys())   # 情感词索引，键
    #     print("sen_loc:", sen_loc)
    #     not_loc = not_word.keys()
    #     degree_loc = degree_word.keys()
    #     senloc = -1
    #     # notloc = -1
    #     # degreeloc = -1
    #     for i in range(0, len(seg_result)):
    #         print(i)
    #         # 如果该词为情感词
    #         if i in sen_loc:
    #             # loc为情感词位置列表的序号
    #             # senloc += 1
    #             # 直接添加该情感词分数
    #             score += w * float(sen_word[i])
    #             senloc += 1
    #             # print("score = %f" % score)
    #             if senloc < len(sen_loc) - 1:
    #                 # 判断该情感词与下一情感词之间是否有否定词或程度副词
    #                 # j为绝对位置
    #                 for j in range(sen_loc[senloc], sen_loc[senloc + 1]):
    #                     # 如果有否定词
    #                     if j in not_loc:
    #                         w *= -1
    #                     # 如果有程度副词
    #                     elif j in degree_loc:
    #                         w *= float(degree_word[j])
    #         # i定位至下一个情感词
    #         if senloc < len(sen_loc) - 1:
    #             i = sen_loc[senloc + 1]
    #
    #         logging.info((str(id) + '------->  score = {}'.format(score)))
    #
    #     return score

    def score_sent(self, sen_word, not_word, degree_word, seg_result):
        # 权重初始化为0.1
        w = 0.1
        score = 0
        # 遍历分词结果
        for i in range(0, len(seg_result)):
            # 若是程度副词
            if seg_result[i] in degree_word.keys():
                w *= float(degree_word[seg_result[i]])
            # 若是否定词
            elif seg_result[i] in not_word.keys():
                w *= -1
            elif seg_result[i] in sen_word.keys():
                score += float(w) * float(sen_word[seg_result[i]])
                w = 0.1
        return score

    def sql_type(self):
        conf = self.read_conf()
        conn = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8' % (self.user, self.password, self.host, int(self.port), self.db))
        df = pd.read_sql("select * from %s" % (conf.get('configure_db', 'table_name')), conn)
        # dataset = df[[conf.get('configure_content', 'id'), conf.get('configure_content', 'title'),
        #               conf.get('configure_content', 'content')]]
        dataset = df[[conf.get('configure_content', 'id'), conf.get('configure_content', 'content')]]
        # logging.info("**********************************")
        # dataset['task_content'] = dataset[conf.get('configure_content', 'title')] + dataset[
            # conf.get('configure_content', 'content')]
        dataset['task_content'] = dataset[conf.get('configure_content')] + dataset[
            conf.get('configure_content', 'content')]
        # data_list = os.listdir(emotion_path)
        data_list = pd.DataFrame()
        data_list['case_content'] = dataset['task_content']
        # data_list['case_serial'] = dataset['case_serial']
        data_list = data_list.reset_index(drop=True)

        return data_list

    def excel_type(self):
        conf = self.read_conf()
        datafile_path = conf.get('configure_file', 'raw_data_path')
        datafile_name = conf.get('configure_file', 'file_name')
        # dataset = pd.read_csv(datafile_path + datafile_name, encoding='utf8', usecols=[conf.get('configure_content', 'id'), conf.get('configure_content', 'title'), conf.get('configure_content', 'content')])
        dataset = pd.read_excel(datafile_path + datafile_name, usecols=[conf.get('configure_content', 'id'), conf.get('configure_content', 'content'), conf.get('configure_content', 'time')])
        # dataset['task_content'] = dataset[conf.get('configure_content', 'title')].astype(str) + dataset[conf.get('configure_content', 'content')].astype(str)
        dataset['task_content'] = dataset[conf.get('configure_content', 'content')].astype(str)
        excel_data = pd.DataFrame()
        excel_data['id'] = dataset['id']
        excel_data['text'] = dataset['task_content']
        excel_data['time'] = dataset['time']
        # excel_data['case_serial'] = dataset['case_serial']
        excel_data = excel_data.reset_index(drop=True)
        return excel_data

    def get_score(self, data_subset):
        # count time
        start = time.clock()
        conf = self.read_conf()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=r"%slog_12345.log" % conf.get('configure_file', 'res_data_path'),
                            filemode='w')
        logging.info("**********************************")
        data_list = data_subset
        if self.database == "database":
            # df = self.excel_type()
            score_list = []
            sen_dict, not_list, degree_dict = self.words()
            for i in range(len(data_list['case_content'])):
                content = data_list.iloc[i]['case_content']
                word_dict, result = self.emotion_dict(content)
                print("打印++++++++")
                print(word_dict, result)
                sen_word, not_word, degree_word = self.classify_words(word_list=result, sen_dict=sen_dict, not_list=not_list, degree_dict=degree_dict)
                score = self.score_sent(sen_word=sen_word, not_word=not_word, degree_word=degree_word, seg_result=result)
                score_list.append(score)
            data_list['score_list'] = score_list
            conn = create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8' % (self.user, self.password, self.host, int(self.port), self.db))
            try:
                data_list.to_sql(name='sql_result', con=conn, if_exists='append', index=False)
                data_list.info('data_list')
            except Exception as e:
                print(e)
        else:
            score_list = []
            sen_dict, not_list, degree_dict = self.words()
            for i in range(len(data_list['text'])):
                content = data_list.iloc[i]['text']
                word_dict, result = self.emotion_dict(content)
                print("打印++++++++")
                print(word_dict, result)
                sen_word, not_word, degree_word = self.classify_words(word_list=result, sen_dict=sen_dict, not_list=not_list, degree_dict=degree_dict)
                score = self.score_sent(sen_word=sen_word, not_word=not_word, degree_word=degree_word, seg_result=result)
                score_list.append(score)
                print("执行至第{}行".format(i))
            data_list['score_list'] = score_list
            # data_list.to_excel(r"%sexcel_result%s.xlsx" % (conf.get('configure_file', 'res_data_path'), time.time()), encoding='utf-8', index=False)
            data_list.to_excel(r"%s%s" % (conf.get('configure_file', 'res_data_path'), conf.get('configure_file', 'file_name')), encoding='utf-8', index=False)
            logging.info('excel_data')
            end = time.clock()
            spend_time = end - start
            logging.info("评分计算完成。")
            logging.info('程序用时：%.2fs' % spend_time)
            print('程序用时：%.2fs' % spend_time)
            print('End...............')

    def threading_process(self):
        if self.database == "database":
            data = self.sql_type()
        else:
            data = self.excel_type()
        avg_samples = int(np.ceil(len(data) / self.n_jobs))  # 平均每个线程要处理的样本数
        threads = []
        start = 0
        for i in range(self.n_jobs):
            if i == 0:
                start = 0
            else:
                start += avg_samples
            if start + avg_samples > len(data):
                pass
            else:
                end = start + avg_samples
            data_subset = data[start:end]
            # print(data_subset)
            t = threading.Thread(target=self.get_score, args=(data_subset,))
            threads.append(t)
        for i in range(self.n_jobs):
            threads[i].start()  # start threads
        for i in range(self.n_jobs):
            threads[i].join()
