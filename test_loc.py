import test_descriptor
import os
import numpy as np
from tqdm import tqdm, trange

import cv2

len_dataset = 276
class file_reader():
    def __init__(self):
        self.descriptor = test_descriptor.test_desciptor()

    def top_score_check(self, dict, score):
        for idx, val in dict:
            if val < score: return True
        return False

    def score_Classifier(self, candidates, label):
        # print(label, 'top 5: ', top5[:, 0])
        if label in candidates[:, 0]:
            # print('case1 = true')
            return True
        for i in range(0, 5):  # 트루로 줄 앵글 범위.
            if (label + i) > len_dataset:
                if (label + i) - len_dataset in candidates[:, 0]: return True
            if (label - i) < 0:
                if (label - i) + len_dataset in candidates[:, 0]: return True
            if (label - i) in candidates[:, 0]: return True
            if (label + i) in candidates[:, 0]: return True
        return False

    def read_txt(self, top_size):
        path = 'E:/Etri/loc_match/image/'
        data_path = 'E:/Etri/loc_match/homo/'
        iter_cnt = 0
        test_name_list = ['orb', 'sift', 'surf', 'AKAZE', 'fast']  # 'orb', 'sift', 'surf', 'AKAZE', 'fast'
        top5, cnt = {}, {}
        # cnt 초기화
        for test_name in test_name_list:
            cnt[test_name] = {'T': 0, 'F': 0}
        print("matching with Top ", top_size)
        avg_rep, avg_mma = {}, {}
        for test_name in test_name_list:
            avg_rep[test_name] = 0
            avg_mma[test_name] = 0

        for target_name in tqdm(os.listdir(path)):
            # 파일 확장자가 (properties)인 것만 처리
            if target_name.endswith("png"):
                target_number = int(target_name.split('_')[1][0:3])
                iter_cnt += 1
                # print("matching target number : ", target_number)

                for data_name in (os.listdir(data_path)):
                    data_number = int(data_name.split('_')[1][0:3])
                    if data_name.endswith("png") and data_number == target_number:
                        for test_name in test_name_list:
                            mma, rep = self.descriptor.featureMatching(path + target_name, data_path + data_name, test_name=test_name,
                                                       local_matching=True)
                            # res = self.descriptor.featureMatching(path + target_name, data_path + data_name,
                            #                                       test_name=test_name, target_num=data_number)
                            avg_rep[test_name] += rep
                            avg_mma[test_name] += mma

                # for test_name in test_name_list:
                #     if iter_cnt % 10 == 0:
                #         print(
                #             '==== Term ACC[{0}] : avg_rep : {1}, avg_mma : {2}, Total: {3}'.format(test_name,
                #                                                                           avg_rep / iter_cnt,
                #                                                                           avg_mma / iter_cnt,
                #                                                                           iter_cnt))

        for test_name in test_name_list:
            print(
                'Result ACC[{0}] : avg_rep : {1:.2f}, avg_mma : {2:.2f}, Total: {3}'.format(test_name,
                                                                                       avg_rep[test_name] / iter_cnt,
                                                                                       avg_mma[test_name] / iter_cnt,
                                                                                       iter_cnt))


if __name__ == "__main__":
    test = file_reader()
    test.read_txt(top_size=1)
    # test.read_txt(top_size=5)
