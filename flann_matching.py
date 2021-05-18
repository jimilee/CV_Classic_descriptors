import cv2, numpy as np
import os
#미라선배 코드.
def get_testdata(detector, folder_path = './DB/_images/', resize_w=0, resize_h=0):
    input_imgs_kps = []
    input_imgs_descs = []

    # load file list
    file_list = os.listdir(folder_path)
    file_list_png = [file for file in file_list if file.endswith(".png")]

    #new_w, new_h = 690, 394

    for file_name in file_list_png:
        img = cv2.imread(os.path.join(folder_path, file_name))
        #if resize_w != 0 and resize_h != 0:
            #img = cv2.resize(img, dsize=(resize_w, resize_h))   # img resize
            #img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, (650, 350))
        kps, descs = detector.detectAndCompute(img, None)
        input_imgs_kps.append(kps)
        input_imgs_descs.append(descs)

    return input_imgs_kps, input_imgs_descs, file_list_png


def get_mapdata(detector, folder_path = './DB/Euler', crop_start_y=0, crop_start_x=0, crop_end_y=0, crop_end_x=0):

    # load 3d map images
    file_list = os.listdir(folder_path)
    file_list_png = [file for file in file_list if file.endswith(".png")]

    map_groups_kps = []
    map_groups_descs = []

    for file_name in file_list_png:
        # load image
        img = cv2.imread(os.path.join(folder_path, file_name))
        if crop_start_x != 0 and crop_start_y != 0 and crop_end_x != 0 and crop_end_y != 0:
            #img = img[crop_start_y:crop_start_x, crop_end_y:crop_end_x]
            img = img[130:img.shape[0]-130, 280:img.shape[1]-280].copy()
            img = cv2.resize(img, (650, 350))
            #img = cv2.GaussianBlur(img, (5, 5), 0)

        # extract key point & feature desctiptor
        kps, descs = detector.detectAndCompute(img, None)
        map_groups_kps.append(kps)
        map_groups_descs.append(descs)

    return map_groups_kps, map_groups_descs, file_list_png

def get_matcher(mode, search):

    matcher = None
    if mode == 'sift':
        bf = cv2.NORM_L2
        index = dict(algorithm=0, trees = 5)
        matcher = cv2.FlannBasedMatcher(index, search)

    elif mode =='akaze':
        index = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        # self.bf = cv2.BFMatcher(bf)
        matcher = cv2.FlannBasedMatcher(index, search)
    elif mode == 'surf':
        bf = cv2.NORM_L2
        index = dict(algorithm=0, trees = 5)
        matcher = cv2.FlannBasedMatcher(index, search)
    elif mode == 'orb':
        bf = cv2.NORM_HAMMING
        index = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        # self.bf = cv2.BFMatcher(bf)
        matcher = cv2.FlannBasedMatcher(index, search)
    # elif key == 'fast':
    #     kp = self.fast.detect(img)

    # if des.type != cv2.CV_32F:
    #     des = des.convertTo(cv2.CV_32F)
    return matcher

def get_detector(mode):

    if mode == 'akaze':
        detector = cv2.AKAZE_create(nOctaves=4, nOctaveLayers=4)
    elif mode == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    elif mode == 'surf':
        detector = cv2.xfeatures2d.SURF_create(1000)
    elif mode == 'orb':
        detector = cv2.ORB_create(nfeatures =1000)
    # elif mode == 'fast':
    #     kp = self.fast.detect(img)

    # if des.type != cv2.CV_32F:
    #     des = des.convertTo(cv2.CV_32F)
    return detector

def matching(matcher,
             input_kps,
             input_descs,
             maps_kps, maps_descs,
             dis_feature_th=80, input_img=None, map_imgs=None):  # 4변수 numpy로 전달

    maps_res = np.zeros(len(maps_kps), dtype=int)

    for i, one_map_descs in enumerate(maps_descs):

        matches = matcher.match(input_descs, one_map_descs)

        good = [m for m in matches if m.distance < dis_feature_th]  # akaze 80

        maps_res[i] = len(good)
        #maps_res[i] = np.mean(good)

        '''if input_img != None and map_imgs != None:
            res = cv2.drawMatches(input_img, in_kps, map_imgs[i], maps_kps[i], good, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow('result', res)
            cv2.waitKey()
            cv2.destroyAllWindows()'''

        #print("good/candidate: %d/%d, img_idx: %s" % (len(good), len(matches), i))

    return maps_res

def alltestfile_matching(matcher,
                         input_imgs_kps_ndarr, input_imgs_descs_ndarr,
                         maps_kps_ndarr, maps_descs_ndarr,
                         in_img_names, maps_name_list, matching_th=80):

    testfile_cnt = len(input_imgs_kps_ndarr)

    accuracy_top5 = 0
    accuracy_top3 = 0
    accuracy_top1 = 0

    map_names = [name.split('_')[0] for name in maps_name_list]

    for i in range(testfile_cnt):

        maps_res = matching(matcher,
                            input_imgs_kps_ndarr[i],
                            input_imgs_descs_ndarr[i],
                            maps_kps_ndarr,
                            maps_descs_ndarr,
                            matching_th)

        argsorted_res = np.argsort(maps_res)[::-1]

        matching_top5 = np.array([map_names[idx] for idx in argsorted_res[:5]], dtype=int)  # top-5
        matching_top3 = np.array([map_names[idx] for idx in argsorted_res[:3]], dtype=int)  # top-3
        matching_top1 = np.array([map_names[idx] for idx in argsorted_res[:1]], dtype=int)  # top-1

        gt_img_name = int(in_img_names[i][6:9])

        if np.any(matching_top5 == gt_img_name):  # if np.any(np.abs(matching_top5 - gt_img_name) < 2):
            accuracy_top5 = accuracy_top5 + 1

        if np.any(matching_top3 == gt_img_name):  # if np.any(np.abs(matching_top3 - gt_img_name) < 2):
            accuracy_top3 = accuracy_top3 + 1

        if np.any(matching_top1 == gt_img_name):  # if np.any(np.abs(matching_top1 - gt_img_name) < 2):
            accuracy_top1 = accuracy_top1 + 1

        print("test img: %s --matching--> 5: %d, %d, %d, %d, %d" % (
        in_img_names[i], matching_top5[0], matching_top5[1], matching_top5[2], matching_top5[3], matching_top5[4]))

    accuracy_top5 = accuracy_top5 / testfile_cnt
    accuracy_top3 = accuracy_top3 / testfile_cnt
    accuracy_top1 = accuracy_top1 / testfile_cnt

    return accuracy_top5, accuracy_top3, accuracy_top1


if __name__ == "__main__":
    print("version:", cv2.__version__)

    mode = 'orb'
    detector = get_detector(mode)
    matcher = get_matcher(mode, dict(checks = 128))

    print("loading input files....")
    input_kps_list, input_descs_list, input_name_list = get_testdata(detector, 'E:/Etri/_images/', 690, 394)

    print("loading map files & extracting key point & feature....")
    maps_kp_list, map_desc_list, maps_name_list = get_mapdata(detector, 'E:/Etri/Euler/', 184, 579, 278, 969)

    input_imgs_kps_ndarr = np.array(input_kps_list)
    input_imgs_descs_ndarr = np.array(input_descs_list)
    maps_kps_ndarr = np.array(maps_kp_list)
    maps_descs_ndarr = np.array(map_desc_list)

    print("matching....")
    acc_top5, acc_top3, acc_top1 = alltestfile_matching(matcher,
                                                        input_imgs_kps_ndarr,
                                                        input_imgs_descs_ndarr,
                                                        maps_kps_ndarr,
                                                        maps_descs_ndarr,
                                                        input_name_list,
                                                        maps_name_list,
                                                        70)

    print("Top5 accuracy =", acc_top5)
    print("Top3 accuracy =", acc_top3)
    print("Top1 accuracy =", acc_top1)


