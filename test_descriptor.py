import numpy as np
import cv2

class test_desciptor():
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures =1000)
        self.surf = cv2.xfeatures2d.SURF_create(1000)
        self.AKAZE = cv2.AKAZE_create()
        self.fast = cv2.FastFeatureDetector_create()
        # self.fast = cv2.FastFeatureDetector.create()
        self.orb = cv2.ORB_create(nfeatures =1000)
        self.br = cv2.BRISK_create()
        self.matcher = None
        self.th_score = 0
        self.search = dict(checks = 128)



    def make_kp(self, img, d_name = None):

        if d_name == 'sift':
            kp = self.sift.detectAndCompute(img, None)
            index = dict(algorithm=0, trees = 5)
            # self.matcher = cv2.BFMatcher()
            self.matcher = cv2.FlannBasedMatcher(index, self.search) #cv2.NORM_L2, crossCheck = False cv2.NORM_L2, crossCheck = False
            self.th_score = 300

        elif d_name == 'AKAZE':
            kp, des =self.AKAZE.detectAndCompute(img, None)
            index = dict(algorithm=6, table_number=5, key_size=10, multi_probe_level=1)
            self.matcher = cv2.FlannBasedMatcher(index, self.search)
            self.th_score = 50

        elif d_name == 'fast':
            kp = self.fast.detect(img, None)
            kp, des = self.br.compute(img, kp)
            self.matcher = cv2.BFMatcher()
            #self.matcher = cv2.FlannBasedMatcher(index, self.search)  #cv2.NORM_L1, crossCheck = Falsecv2.NORM_L2, crossCheck = True
            self.th_score = 300

        elif d_name == 'surf':
            kp, des = self.sift.detectAndCompute(img, None)
            index = dict(algorithm=0, trees=5)
            #self.matcher = cv2.BFMatcher()
            self.matcher = cv2.FlannBasedMatcher(index, self.search)  #cv2.NORM_L1, crossCheck = Falsecv2.NORM_L2, crossCheck = True
            self.th_score = 300
        elif d_name == 'orb':
            kp, des = self.orb.detectAndCompute(img, None)
            index = dict(algorithm=6, table_number=5, key_size=10, multi_probe_level=1)
            self.matcher = cv2.FlannBasedMatcher(index, self.search)
            self.th_score = 70
        # elif key == 'fast':
        #     kp = self.fast.detect(img)

        # if des.type != cv2.CV_32F:
        #     des = des.convertTo(cv2.CV_32F)
        return kp, des

    def kp_matcher(self, matcher, des1, des2, d_name = None):
        good, matches = [],[]
        if d_name == 'sift':
            try:
                matches = matcher.knnMatch(des1, des2, k=2)
                if len(matches) == 0: return [], []
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return [], []

        elif d_name == 'AKAZE':
            matches = matcher.knnMatch(des1, des2, k=2)
            if len(matches) == 0: return [], []
            try:
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return [], []

        elif d_name == 'fast':
            matches = matcher.knnMatch(des1, des2, k=2)
            if len(matches) == 0: return [], []
            try:
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return [], []

        elif d_name == 'surf':
            try:
                matches = matcher.knnMatch(des1, des2, k=2)
                if len(matches) == 0: return [], []
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return [], []
        elif d_name == 'orb':
            try:
                matches = matcher.knnMatch(des1, des2, k=2)
                if len(matches) == 0: return [], []
                good = [m for m, n in matches if m.distance < 0.8 * n.distance]
            except:
                return [], []
            # matches = matcher.match(des1, des2)
            # print(len(matches))
            # if len(matches) == 0: return 0, 0
            # else: good = [m for m in matches if m.distance < self.th_score]

        return good, matches

    def featureMatching(self, img1_path, img2_path, test_name, target_num=0):
        k_size = 3
        src1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        src2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        croped = src2[130:src1.shape[0]-130, 270:src1.shape[1]-270].copy()

        img1 = cv2.resize(src1, (720, 480))
        img2 = cv2.resize(croped, (720, 480))
        # img1 = cv2.blur(rsize1, ksize=(k_size, k_size))
        # img2 = cv2.blur(rsize2, ksize=(k_size, k_size))
        #그리드 매칭 추가
        img_h, img_w = img1.shape

        total_matches = 0
        total_good = 0
        grid_size = [480, 720]

        for w in range(0, img_w, 690):
            for h in range(0, img_h, 360):
                if h+grid_size[0] > img_h or w+grid_size[1] > img_w:
                    break
                croped1 = img1[h:h + grid_size[0], w:w + grid_size[1]].copy()
                croped2 = img2[h:h + grid_size[0], w:w + grid_size[1]].copy()

                kp1, des1 = self.make_kp(croped1, test_name)
                kp2, des2 = self.make_kp(croped2, test_name)

                if len(kp1) == 0 or len(kp2) == 0 : continue

                # matches = self.matcher.match(des1, des2)
                # if len(matches) == 0: continue
                #
                # dist = [_.distance for _ in matches]
                # avg = sum(dist, 0.0)/len(dist)
                # print('avg : ', avg)

                good, matches = self.kp_matcher(self.matcher, des1, des2, test_name)
                if len(good) == 0: continue
                # # # 확인용 출력.
                # if target_num%80==0 :
                #     res = cv2.drawMatches(croped1, kp1, croped2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                #     cv2.imshow('res', res)
                #     # cv2.imshow('res2', croped2)
                #     cv2.waitKey()
                #     cv2.destroyWindow('res')
                #     # cv2.destroyWindow('res2')
                #     print(target_num,' : ', test_name, '/ ', len(good) / len(matches),
                #           'good : {0} / {1}.'.format(len(good), len(matches)))

                total_good+=len(good)
                total_matches+=len(matches)
                # print(test_name, '/ ',len(good) / len(matches), 'good : {0} / {1}.'.format(len(good), len(matches)))


        # blured = cv2.blur(img1, ksize=(k_size,k_size))
        #
        #
        # croped = src2[200:550, 300:950].copy()
        # blured2 = cv2.blur(croped, ksize=(k_size, k_size))


        # kp1, des1 = self.make_kp(blured, test_name)
        # kp2, des2 = self.make_kp(blured2, test_name)

        # bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
        # bf = cv2.BFMatcher(cv2.NORM_L2)
        # matches = bf.match(des1, des2)

        # matches = self.bf.knnMatch(des1, des2, k=2)
        # good = [m for m, n in matches if m.distance < 0.7*n.distance]
        # res = cv2.drawMatches(img1, kp1, croped, kp2, good, res, flags=0)

        if total_matches == 0 : return 0.
        return (total_good/total_matches)*100


# descriptor = test_desciptor()
# descriptor.featureMatching('test1.png','test2.png', 'orb')
# descriptor.featureMatching('test1.png','test2.png', 'sift')
# descriptor.featureMatching('test1.png','test2.png', 'surf')
# cv2.destroyWindow()