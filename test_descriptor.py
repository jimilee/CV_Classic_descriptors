import numpy as np
import cv2

class test_desciptor():
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures =500)
        self.surf = cv2.xfeatures2d.SURF_create(500)
        self.AKAZE = cv2.AKAZE_create()
        self.fast = cv2.FastFeatureDetector_create()
        self.orb = cv2.ORB_create(nfeatures =500)
        self.br = cv2.BRISK_create()
        self.matcher = None
        self.th_score = 0
        self.search = dict(checks = 128)

    def make_kp(self, img, d_name = None):#각 디스크립터별 세부사항 세팅
        kp, des = None, None
        if d_name == 'sift':
            kp, des = self.sift.detectAndCompute(img, None)
            index = dict(algorithm=0, trees=5)
            self.matcher = cv2.BFMatcher()
            #self.matcher = cv2.FlannBasedMatcher(index,self.search)  # cv2.NORM_L1, crossCheck = Falsecv2.NORM_L2, crossCheck = True
            self.th_score = 300

        elif d_name == 'AKAZE':
            kp, des =self.AKAZE.detectAndCompute(img, None)
            index = dict(algorithm=6, table_number=5, key_size=10, multi_probe_level=1)
            self.matcher = cv2.FlannBasedMatcher(index, self.search)
            self.th_score = 50

        elif d_name == 'fast':
            kp = self.fast.detect(img, None)
            kp, des = self.br.compute(img, kp)
            index = dict(algorithm=0, trees=5)
            #self.matcher = cv2.BFMatcher()
            self.matcher = cv2.FlannBasedMatcher(index, self.search)  #cv2.NORM_L1, crossCheck = Falsecv2.NORM_L2, crossCheck = True
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
        if kp is None or des is None: print(d_name)
        return kp, des

    def kp_matcher(self, matcher, des1, des2, d_name = None): #각 디스크립터별 매칭방식 세팅
        good, matches = [],[]
        if d_name == 'sift':
            try:
                matches = matcher.knnMatch(des1, des2, k=2)
                if len(matches) == 0: return good, matches
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return good, matches

        elif d_name == 'AKAZE':
            matches = matcher.knnMatch(des1, des2, k=2)
            if len(matches) == 0: return good, matches
            try:
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return good, matches

        elif d_name == 'fast':
            matches = matcher.knnMatch(des1, des2, k=2)
            if len(matches) == 0: return good, matches
            try:
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return good, matches

        elif d_name == 'surf':
            try:
                matches = matcher.knnMatch(des1, des2, k=2)
                if len(matches) == 0: return good, matches
                good = [m for m,n in matches if m.distance < 0.8 *n.distance]
            except:
                return good, matches
        elif d_name == 'orb':
            try:
                matches = matcher.knnMatch(des1, des2, k=2)
                if len(matches) == 0: return good, matches
                good = [m for m, n in matches if m.distance < 0.8 * n.distance]
            except:
                return good, matches

        return good, matches

    def featureMatching(self, img1_path, img2_path, test_name, target_num=0, show_img = False):
        src1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        src2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        croped = src2[130:src1.shape[0]-130, 270:src1.shape[1]-270].copy()

        img1 = cv2.resize(src1, (720, 480))
        img2 = cv2.resize(croped, (720, 480))
        # img1 = cv2.blur(rsize1, ksize=(k_size, k_size))
        # img2 = cv2.blur(rsize2, ksize=(k_size, k_size))
        #윈도우 슬라이싱 매칭 사이즈.
        img_h, img_w = img1.shape

        total_matches = 0
        total_good = 0
        window_size = [img_h, img_w]

        for w in range(0, img_w, window_size[1]-1):
            for h in range(0, img_h, window_size[0]-1):
                if h+window_size[0] > img_h or w+window_size[1] > img_w:
                    break
                croped1 = img1[h:h + window_size[0], w:w + window_size[1]].copy()
                croped2 = img2[h:h + window_size[0], w:w + window_size[1]].copy()

                kp1, des1 = self.make_kp(croped1, test_name)
                kp2, des2 = self.make_kp(croped2, test_name)

                if len(kp1) == 0 or len(kp2) == 0 : continue

                good, matches = self.kp_matcher(self.matcher, des1, des2, test_name)
                if len(good) == 0: continue
                # # 확인용 출력.
                if target_num%80==0 and show_img:
                    res = cv2.drawMatches(croped1, kp1, croped2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow('res', res)
                    # cv2.imshow('res2', croped2)
                    cv2.waitKey()
                    cv2.destroyWindow('res')
                    # cv2.destroyWindow('res2')
                    print(target_num,' : ', test_name, '/ ', len(good) / len(matches),
                          'good : {0} / {1}.'.format(len(good), len(matches)))

                total_good+=len(good)
                total_matches+=len(matches)
                # print(test_name, '/ ',len(good) / len(matches), 'good : {0} / {1}.'.format(len(good), len(matches)))


        if total_matches == 0 : return 0.
        return (total_good/total_matches)*100

# descriptor = test_desciptor()
# descriptor.featureMatching('test1.png','test2.png', 'orb')
# descriptor.featureMatching('test1.png','test2.png', 'sift')
# descriptor.featureMatching('test1.png','test2.png', 'surf')
# cv2.destroyWindow()