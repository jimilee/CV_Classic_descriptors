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
        path = 'E:/Etri/_images/'
        data_path = 'E:/Etri/Euler/'
        iter_cnt = 0
        test_name_list = ['orb', 'sift', 'surf', 'AKAZE', 'fast']  # 'orb', 'sift', 'surf', 'AKAZE', 'fast'
        top5, cnt = {}, {}
        # cnt 초기화
        for test_name in test_name_list:
            cnt[test_name] = {'T': 0, 'F': 0}
        print("matching with Top ", top_size)
        for target_name in tqdm(os.listdir(path)):
            # 파일 확장자가 (properties)인 것만 처리
            if target_name.endswith("png"):
                target_number = int(target_name.split('_')[1][0:3])
                iter_cnt += 1
                # print("matching target number : ", target_number)

                # Top5 초기화
                for test_name in test_name_list:
                    top5[test_name] = np.zeros((top_size, 2))

                for data_name in (os.listdir(data_path)):
                    data_number = int(data_name.split('_')[0])
                    if data_name.endswith("png"):
                        for test_name in test_name_list:
                            res = self.descriptor.featureMatching(path + target_name, data_path + data_name,
                                                                  test_name=test_name, target_num=data_number)
                            if self.top_score_check(top5[test_name], res):
                                top5[test_name][np.argmin(top5[test_name][:, 1])] = np.array([data_number, res])

                for test_name in test_name_list:
                    if self.score_Classifier(top5[test_name], target_number):
                        cnt[test_name]['T'] += 1
                    else:
                        cnt[test_name]['F'] += 1

                    if iter_cnt % 10 == 0:
                        print(
                            '==== Term ACC[{0}] : {1}, T: {2}, F: {3}, Total: {4}'.format(test_name,
                                                                                          cnt[test_name]['T'] / (cnt[test_name]['T']+cnt[test_name]['F']),
                                                                                          cnt[test_name]['T'],
                                                                                          cnt[test_name]['F'],
                                                                                          iter_cnt))

        for test_name in test_name_list:
            print('Result ACC[{0}] : {1}, T: {2}, F: {3}, Total: {4}. '.format(test_name,
                                                                               cnt[test_name]['T'] / (cnt[test_name]['T']+cnt[test_name]['F']),
                                                                               cnt[test_name]['T'],
                                                                               cnt[test_name]['F'],
                                                                               iter_cnt))


if __name__ == "__main__":
    test = file_reader()
    test.read_txt(top_size=1)
    test.read_txt(top_size=5)
