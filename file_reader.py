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
        if label in top5[:,0]:
            print('case1 = true')
            return True
        for i in range(0,2): #트루로 줄 앵글 범위.
            if (label+i) > 276:
                if (label+i) - 276 in top5[:,0]: return True
            if (label-i) < 0:
                if (label-i) + 276 in top5[:,0]: return True
            if (label-i) in top5[:,0]: return True
            if (label+i) in top5[:,0]: return True
        return False

    def read_txt(self):
        path = 'E:/Etri/_images/'
        data_path = 'E:/Etri/Euler/'
        Total = 0
        cnt_surf = {'T':0, 'F':0}
        cnt_sift = {'T': 0, 'F': 0}
        cnt_orb = {'T': 0, 'F': 0}
        cnt_akaze = {'T': 0, 'F': 0}
        cnt_fast = {'T': 0, 'F': 0}

        for target_name in os.listdir(path):
            # 파일 확장자가 (properties)인 것만 처리
            if target_name.endswith("png"):
                target_number = int(target_name.split('_')[1][0:3])
                Total+=1
                print("matching target number : ", target_number)

                #Top5 초기화
                top5_surf, top5_sift, top5_orb, top5_akaze, top5_fast = np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))

                for data_name in os.listdir(data_path):
                    data_number = int(data_name.split('_')[0])
                    if data_name.endswith("png"):

                        res_orb = self.descriptor.featureMatching(path + target_name, data_path + data_name,
                                                                        test_name='orb', target_num = data_number)
                        if self.top5_check(top5_orb, res_orb):
                            top5_orb[np.argmin(top5_orb[:,1])] = np.array([data_number, res_orb])

                        # res_sift = self.descriptor.featureMatching(path + target_name, data_path + data_name,
                        #                                                 test_name='sift', target_num = data_number)
                        # if self.top5_check(top5_sift, res_sift):
                        #     top5_sift[np.argmin(top5_sift[:,1])] = np.array([data_number, res_sift])
                        #
                        # res_surf = self.descriptor.featureMatching(path + target_name, data_path+data_name,
                        #                                                 test_name='surf', target_num = data_number)
                        # if self.top5_check(top5_surf, res_surf):
                        #     top5_surf[np.argmin(top5_surf[:,1])] = np.array([data_number, res_surf])

                        res_akaze = self.descriptor.featureMatching(path + target_name, data_path+data_name,
                                                                        test_name='AKAZE', target_num = data_number)
                        if self.top5_check(top5_akaze, res_akaze):
                            top5_akaze[np.argmin(top5_akaze[:,1])] = np.array([data_number, res_akaze])
                        #
                        # res_fast = self.descriptor.featureMatching(path + target_name, data_path + data_name,
                        #                                           test_name='fast', target_num=data_number)
                        # if self.top5_check(top5_fast, res_orb):
                        #     top5_fast[np.argmin(top5_fast[:, 1])] = np.array([data_number, res_fast])
                # res = self.top5_Classifier(top5_orb, target_number)
                # print(res)
                if self.top5_Classifier(top5_orb, target_number): cnt_orb['T'] += 1
                else : cnt_orb['F']+=1
                if self.top5_Classifier(top5_sift, target_number): cnt_sift['T'] += 1
                else : cnt_sift['F']+=1
                if self.top5_Classifier(top5_surf, target_number): cnt_surf['T'] += 1
                else : cnt_surf['F']+=1
                if self.top5_Classifier(top5_akaze, target_number): cnt_akaze['T'] += 1
                else : cnt_akaze['F']+=1
                if self.top5_Classifier(top5_fast, target_number): cnt_fast['T'] += 1
                else : cnt_fast['F']+=1
                if Total%10==0:
                    print(
                        '==== Terms ACC[orb] : {0}, T: {1}, F: {2}, Total: {3}.'.format(cnt_orb['T'] / Total, cnt_orb['T'],
                                                                                     cnt_orb['F'],
                                                                                     Total))
                    print('==== Terms ACC[surf] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_surf['T'] / Total,
                                                                                        cnt_surf['T'], cnt_surf['F'],
                                                                                        Total))

                    print(
                        '==== Terms ACC[sift] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_sift['T'] / Total,
                                                                                      cnt_sift['T'], cnt_sift['F'],
                                                                                      Total))
                    print(
                        '==== Terms ACC[AKAZE] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_akaze['T'] / Total,
                                                                                      cnt_akaze['T'], cnt_akaze['F'],
                                                                                      Total))
                    print(
                        '==== Terms ACC[fast] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_fast['T'] / Total,
                                                                                           cnt_fast['T'], cnt_fast['F'],
                                                                                           Total))

        print('Result ACC[surf] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_surf['T']/Total, cnt_surf['T'], cnt_surf['F'], Total))

        print(
            'Result ACC[sift] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_sift['T'] / Total, cnt_sift['T'], cnt_sift['F'],
                                                                    Total))
        print(
            'Result ACC[orb] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_orb['T'] / Total, cnt_orb['T'], cnt_orb['F'],
                                                                    Total))
        print(
            'Result ACC[AKAZE] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_akaze['T'] / Total,
                                                                               cnt_akaze['T'], cnt_akaze['F'],
                                                                               Total))
        print(
            'Result ACC[fast] : {0}, T: {1}, F: {2}, Total: {3}. '.format(cnt_fast['T'] / Total,
                                                                              cnt_fast['T'], cnt_fast['F'],
                                                                              Total))

test = file_reader()
test.read_txt()