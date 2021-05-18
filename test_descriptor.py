import numpy as np
import cv2

# 지역 기술자 매칭 알고리즘 및 전역 기술자 평가 코드
class test_desciptor():
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures =300) #nfeatures =500
        self.surf = cv2.xfeatures2d.SURF_create(300)
        self.AKAZE = cv2.AKAZE_create()
        self.fast = cv2.FastFeatureDetector_create(300)
        self.orb = cv2.ORB_create(nfeatures =300) #nfeatures =500
        self.br = cv2.BRISK_create()
        self.matcher = None
        self.th_score = 0
        self.search = dict(checks = 128)

    # 각 디스크립터별 세부사항 세팅, matcher 방식 선택. BFMatcher 및 FlannBasedMatcher
    def make_kp(self, img, d_name = None):
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
            #self.matcher = cv2.FlannBasedMatcher(index, self.search)
            self.matcher = cv2.BFMatcher()
            self.th_score = 50

        elif d_name == 'fast':
            kp = self.fast.detect(img, None)
            kp, des = self.br.compute(img, kp)
            index = dict(algorithm=6, table_number=5, key_size=10, multi_probe_level=1)
            self.matcher = cv2.BFMatcher()
            #self.matcher = cv2.FlannBasedMatcher(index, self.search)  #cv2.NORM_L1, crossCheck = Falsecv2.NORM_L2, crossCheck = True
            self.th_score = 300

        elif d_name == 'surf':
            kp, des = self.sift.detectAndCompute(img, None)
            index = dict(algorithm=0, trees=5)
            self.matcher = cv2.BFMatcher()
            #self.matcher = cv2.FlannBasedMatcher(index, self.search)  #cv2.NORM_L1, crossCheck = Falsecv2.NORM_L2, crossCheck = True
            self.th_score = 300

        elif d_name == 'orb':
            kp, des = self.orb.detectAndCompute(img, None)
            index = dict(algorithm=6, table_number=5, key_size=10, multi_probe_level=1)
            self.matcher = cv2.BFMatcher()
            #self.matcher = cv2.FlannBasedMatcher(index, self.search)
            self.th_score = 70
        if kp is None or des is None: print(d_name)
        return kp, des

    # 키포인트가 두 이미지에서 동일한 위치에 나오는지 확인
    def compute_repeatability(self, kp1, kp2, th=3):
        kp1 = cv2.KeyPoint_convert(kp1)
        kp2 = cv2.KeyPoint_convert(kp2)
        repeatability = 0
        N1 = len(kp1)
        N2 = len(kp2)
        if N2 != 0 and N1 != 0:
            kp1 = np.expand_dims(kp1, 1)
            kp2 = np.expand_dims(kp2, 0)

            norm = np.linalg.norm(kp1 - kp2, ord=None, axis=-1)
            cnt1 = np.sum(np.min(norm, axis=1) <= th)
            cnt2 = np.sum(np.min(norm, axis=0) <= th)

            repeatability = (cnt1 + cnt2) / (N1 + N2)
        return repeatability

    # 키포인트가 두 이미지에서 실제 매칭된 위치가 맞는지
    def compute_mma(self, kp1, kp2, good, th=3):
        kp1 = cv2.KeyPoint_convert(kp1)
        kp2 = cv2.KeyPoint_convert(kp2)

        src_pts = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx] for m in good]).reshape(-1, 2)

        N1 = src_pts.shape[0]
        N2 = dst_pts.shape[0]
        matching_score = 0
        if N2 != 0 and N1 != 0:
            norm = np.linalg.norm(src_pts - dst_pts, ord=None, axis=1)
            is_match = norm <= th
            match_n = np.sum(is_match)
            matching_score = match_n / len(norm)
        return matching_score

    # 각 디스크립터별 매칭방식 세팅
    def kp_matcher(self, matcher, des1, des2, d_name = None):
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

    # 특징 디스크립터를 이용한 매칭 알고리즘
    def featureMatching(self, img1_path, img2_path, test_name, target_num=0, show_img = False, local_matching = False):
        src1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        src2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if not local_matching:
            croped = src2[130:src1.shape[0]-130, 270:src1.shape[1]-270].copy()

            img1 = cv2.resize(src1, (640, 480))
            img2 = cv2.resize(croped, (640, 480))
        else:
            img1 = src1
            img2 = src2

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
                if local_matching and not show_img:

                    return self.compute_mma(kp1, kp2, good), self.compute_repeatability(kp1, kp2)

                if len(good) == 0: continue
                # # 확인용 출력.
                if show_img:
                    res = cv2.drawMatches(croped1, kp1, croped2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow(test_name, res)
                    cv2.waitKey()
                    cv2.destroyWindow(test_name)
                    print(target_num,' : ', test_name, '/ ', len(good) / len(matches),
                          'good : {0} / {1}.'.format(len(good), len(matches)))
                    print('compute_repeatability.', self.compute_repeatability(kp1, kp2))
                    print('compute_mma.', self.compute_mma(kp1, kp2, good))
                total_good+=len(good)
                total_matches+=len(matches)


        if total_matches == 0 : return 0.
        return (total_good/total_matches)*100


if __name__ == "__main__":
    descriptor = test_desciptor()
    descriptor.featureMatching('homo_000.png','image_000.png', 'fast', show_img=True, local_matching=True)
