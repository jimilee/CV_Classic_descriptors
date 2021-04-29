import test_descriptor
import os
import numpy as np
import cv2


class file_reader():
    def __init__(self):
        self.descriptor = test_descriptor.test_desciptor()

    def top5_check(self, dict, score):
        for idx, val in dict:
            if val < score: return True
        return False

    def top5_Classifier(self, top5, label):
        print(label, 'top 5: ', top5[:, 0])
        if label in top5[:, 0]:
            print('case1 = true')
            return True
        for i in range(0, 2):  # 트루로 줄 앵글 범위.
            if (label + i) > 276:
                if (label + i) - 276 in top5[:, 0]: return True
            if (label - i) < 0:
                if (label - i) + 276 in top5[:, 0]: return True
            if (label - i) in top5[:, 0]: return True
            if (label + i) in top5[:, 0]: return True
        return False

    def read_txt(self):
        path = 'E:/Etri/_images/'
        data_path = 'E:/Etri/Euler/'
        iter_cnt = 0
        test_name_list = ['orb']  # 'orb', 'sift', 'surf', 'AKAZE', 'fast'
        top5, cnt = {}, {}
        scorebox = 1
        for target_name in os.listdir(path):
            # 파일 확장자가 (properties)인 것만 처리
            if target_name.endswith("png"):
                target_number = int(target_name.split('_')[1][0:3])
                iter_cnt += 1
                print("matching target number : ", target_number)

                # Top5 초기화
                for test_name in test_name_list:
                    top5[test_name] = np.zeros((scorebox, 2))
                    cnt[test_name] = {'T': 0, 'F': 0}
                # top5_surf, top5_sift, top5_orb, top5_akaze, top5_fast = np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))

                for data_name in os.listdir(data_path):
                    data_number = int(data_name.split('_')[0])
                    if data_name.endswith("png"):
                        for test_name in test_name_list:
                            res = self.descriptor.featureMatching(path + target_name, data_path + data_name,
                                                                  test_name=test_name, target_num=data_number)
                            if self.top5_check(top5[test_name], res):
                                top5[test_name][np.argmin(top5[test_name][:, 1])] = np.array([data_number, res])

                for test_name in test_name_list:
                    if self.top5_Classifier(top5[test_name], target_number):
                        cnt[test_name]['T'] += 1
                    else:
                        cnt[test_name]['F'] += 1

                    if iter_cnt % 10 == 0:
                        print(
                            '==== Term ACC[{0}] : {1}, T: {2}, F: {3}, Total: {4}'.format(test_name,
                                                                                          cnt[test_name]['T'] / (cnt[test_name]['T']+cnt[test_name]['F']),
                                                                                          cnt[test_name]['T'],
                                                                                          cnt[test_name]['F'],
                                                                                          Total))

        for test_name in test_name_list:
            print('Result ACC[{0}] : {1}, T: {2}, F: {3}, Total: {4}. '.format(test_name,
                                                                               cnt[test_name]['T'] / (cnt[test_name]['T']+cnt[test_name]['F']),
                                                                               cnt[test_name]['T'],
                                                                               cnt[test_name]['F'],
                                                                               Total))


def __main__():
    test = file_reader()
    test.read_txt()
