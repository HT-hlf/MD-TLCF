# coding:utf-8


from mmdet.apis import init_detector, inference_detector
import os
import mmcv
import numpy as np
import cv2
from mmdet.core.visualization import imshow_det_bboxes_ht_rgb,imshow_det_bboxes_ht_rgb_noshow

from filterpy.kalman import KalmanFilter
np.random.seed(0)

def create_dir(save_crop_miner_path):
    if not os.path.isdir(save_crop_miner_path):
        os.makedirs(save_crop_miner_path)

def ht_find_depth_encode_decode(img_find_depth, bbox, n_compress=5,middle_the_x_ratio=0.25,middle_the_y_ratio=0.25):
    xmin, ymin, xmax, ymax = bbox

    x_l = xmax - xmin
    y_l = ymax - ymin
    x_m = int(middle_the_x_ratio * x_l)
    y_m = int(middle_the_y_ratio * y_l)
    depth_image_data_1_middle = img_find_depth[
                              int((ymin + ymax - 2) / 2) - y_m:int((ymin + ymax - 2) / 2) + y_m + 1,
                              int((xmin + xmax - 2) / 2) - x_m:int((xmin + xmax - 2) / 2) + x_m + 1,
                              0]


    depth_image_list = depth_image_data_1_middle.flatten()
    depth_image_data_1_middle_height = depth_image_list * (780 / 255)
    depth_image_data_1_middle_height = depth_image_data_1_middle_height[depth_image_data_1_middle_height >= 50]
    # depth_image_data_1_middle_height = depth_image_data_1_middle_height[depth_image_data_1_middle_height <= 780]
    depth_image_data_1_middle_height_compress = depth_image_data_1_middle_height//n_compress
    depth_image_data_1_middle_height_compress = depth_image_data_1_middle_height_compress.astype(np.uint32)

    if len(depth_image_data_1_middle_height_compress) > 0:
      counts = np.bincount(depth_image_data_1_middle_height_compress)
      mean = np.argmax(counts)
      sigma = (0.5*780/n_compress) ** 2
      mean = mean*n_compress
    else:
      mean = np.nan

      sigma = 160000
    return mean


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))



def iou_batch(bb_test, bb_gt):

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):

  detections = detections[0:5]
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)


  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def convert_bbox_to_z(bbox):

  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):

  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
def detection_to_trackers_height_associate(distance,trackers,distance_thre=0.2):
    if np.isnan(distance):
        return False
    for tracker in trackers:
        if abs(tracker[4]-distance)<=distance_thre:
            return True
    return False

def detection_to_trackers_iou_associate(detections,trackers,iou_threshold = 0.3):

  for tracker in trackers:
      bi = [max(detections[0], tracker[0]), max(detections[1], tracker[1]), min(detections[2], tracker[2]), min(detections[3],tracker[3])]
      iw = bi[2] - bi[0] + 1
      ih = bi[3] - bi[1] + 1
      if iw > 0 and ih > 0:
          ua = (detections[2] - detections[0] + 1) * (detections[3] - detections[1] + 1) + (tracker[2] - tracker[0]
                                                                + 1) * (tracker[3] - tracker[1] + 1) - iw * ih
          iou = iw * ih / ua
          if iou >= iou_threshold:
              return True
          else:
              return False
      else:
          return False




class KalmanBoxTracker(object):

    count = 0


    def __init__(self, bbox):

        self.kf = KalmanFilter(dim_x=9, dim_z=5)

        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0]])
        self.kf.R[2:4, 2:4] *= 10.
        self.kf.P[5:8, 5:8] *= 1000.
        self.kf.P[8, 8] *= 1000
        self.kf.P *= 10.
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:8, 5:8] *= 0.01
        self.kf.Q[8, 8] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.kf.x[4] = bbox[4]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        x_y_s_r = convert_bbox_to_z(bbox)
        self.kf.update([np.append(x_y_s_r, bbox[4])])

    def predict(self):
        if ((self.kf.x[7] + self.kf.x[2]) <= 0):
            self.kf.x[7] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        x_y_s_r = convert_x_to_bbox(self.kf.x)
        self.history.append([np.append(x_y_s_r, self.kf.x[4])])
        return self.history[-1]

    def get_state(self):
        xy_xy = convert_x_to_bbox(self.kf.x)
        return np.array([[xy_xy[0, 0], xy_xy[0, 1], xy_xy[0, 2], xy_xy[0, 3], self.kf.x[4]]])


class ht_tracker(object):
    def __init__(self, max_frame=1, min_hits=3, iou_threshold=0.3):
        self.max_frame = max_frame
        self.min_hits = min_hits

        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, trks, dets=np.empty((0, 6))):
        ret = []
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        # TODO 这里加入深度值判断 大于阈值，选择性加入；小于的立即加入
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            if (trk.time_since_update > self.max_frame):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 6))

    def update_nodet(self):
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            i -= 1

            if (trk.time_since_update > self.max_frame):
                self.trackers.pop(i)

    def predict(self):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        return trks


if __name__ == '__main__':

    config_file = 'config/network_fusing_RGBD.py'
    checkpoint_file = 'model.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')


    conf_thre = 0.5
    depth_w, depth_h = 775, 532
    max_frame = 2
    min_hits = 0
    tracker_iou_threshold = 0.55

    iou_thre = 0.6
    distance_thre = 45
    low_state_num = 3


    raw_root_path = './data_miner/RGBD_r_13_2'

    path = os.path.join(raw_root_path, 'rgb')
    depth_path = os.path.join(raw_root_path, 'depth')

    depth_path_list=os.listdir(depth_path)
    depth_path_list.sort(key=lambda x:int(x.split('.')[0].split('_')[-1]))
    mot_tracker = ht_tracker(max_frame=max_frame,
                       min_hits=min_hits,
                       iou_threshold=tracker_iou_threshold)

    for frame ,filename in enumerate(depth_path_list):

        img_rgb=os.path.join(path,filename)
        img_depth = os.path.join(depth_path,filename)


        result = inference_detector(model, filename, img_prefix_miner=path,
                                    img_prefix_depth_miner=depth_path)

        depth_img = mmcv.imread(img_depth)
        rgb_img = mmcv.imread(img_rgb)

        trust_states = []
        low_conf_states = []
        result_distance = []


        trust_bboxes_depth = []
        trust_bboxes = []
        for bbox in result[0]:
            x1, y1, x2, y2 = int(max(0, min(bbox[0], depth_w - 1))), int(
                max(0, min(bbox[1], depth_h - 1))), int(max(0, min(bbox[2], depth_w - 1))), int(
                max(0, min(bbox[3], depth_h - 1)))
            distance = ht_find_depth_encode_decode(depth_img, bbox[0:4], n_compress=5, middle_the_x_ratio=0.25,
                                                   middle_the_y_ratio=0.25)
            if not np.isnan(distance):
                result_distance.append([x1, y1, x2, y2, distance, bbox[4]])

        low_state_count = 0
        for bbox_distance in result_distance:
            if bbox_distance[5]>=conf_thre:
                trust_states.append(
                    bbox_distance)

                trust_bboxes.append([bbox_distance[0], bbox_distance[1], bbox_distance[2], bbox_distance[3], bbox_distance[5]])
                trust_bboxes_depth.append(bbox_distance[4])

            elif low_state_count < low_state_num:
                low_state_count += 1
                low_conf_states.append(
                        bbox_distance)
            else:
                pass

        trackers = mot_tracker.predict()
        for states in low_conf_states:
            if detection_to_trackers_iou_associate(states[0:4], trackers,iou_threshold=iou_thre) and detection_to_trackers_height_associate(states[4], trackers, distance_thre):
                trust_states.append(states)
                trust_bboxes.append([states[0], states[1], states[2], states[3], states[5]])
                trust_bboxes_depth.append(states[4])

        if len(trust_states) != 0:
            trust_states = np.array(trust_states)
            trust_states = np.reshape(trust_states, (-1, 6))
            mot_tracker.update(trackers, trust_states)
        else:
            mot_tracker.update_nodet()
        show = False
        wait_time = 0
        if len(trust_bboxes)!=0:
            bboxes = np.array(trust_bboxes)
            labels =   [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate([bboxes])]
            labels = np.concatenate(labels)

            img_depth_bbox = imshow_det_bboxes_ht_rgb(
                depth_img,
                trust_bboxes_depth,
                bboxes,
                labels,
                None,
                class_names=('miner',),
                score_thr=0,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=show,
                wait_time=wait_time,
                out_file=None)
            img_rgb_bbox = imshow_det_bboxes_ht_rgb(
                rgb_img,
                trust_bboxes_depth,
                bboxes,
                labels,
                None,
                class_names=('miner',),
                score_thr=0,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=show,
                wait_time=wait_time,
                out_file=None)
        else:

            img_depth_bbox = imshow_det_bboxes_ht_rgb_noshow(depth_img, win_name='',
                                            show=show,
                                            wait_time=wait_time,
                                            out_file=None)
            img_rgb_bbox =imshow_det_bboxes_ht_rgb_noshow(rgb_img, win_name='',
                                            show=show,
                                            wait_time=wait_time,
                                            out_file=None)
        img_depth_bbox_sum = np.hstack((img_rgb_bbox, img_depth_bbox))
        cv2.imshow('img_depth_bbox_sum',img_depth_bbox_sum)
        cv2.waitKey(1)


