# import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
from ml3d.datasets import NuScenes, SemanticKITTI
from  ml3d.vis import Visualizer
import sys 
import numpy as np
import open3d as o3d

script_path = '/home/levin/workspace/ros_projects/src/vslam_localization/scripts'
sys.path.append(script_path)

from utility.poseinfo import PoseInfo

class Simulate3DData(object):
    def __init__(self):
        self.pw = np.array([[6, -2, 0], [9, -2, 0], [7.5, -2, 3]])
        # self.Twb0 = PoseInfo().construct_fromyprt([0, 0, 0], [0, 0, 0])
        # self.Twb1 = PoseInfo().construct_fromyprt([0, 0, 0], [1, 0, 0])
        self.Tbl = PoseInfo().construct_fromyprt([0, 0, 0], [0, 0, 0])
        return
    def gen_data(self, Twb, frame_id):
        Twl = Twb * self.Tbl
        points = self.pw
        pts_4d = np.concatenate(
            [points,
             np.ones((len(points), 1))], axis=-1)
        
        pts_obs = pts_4d.dot(Twl.I.T.T)[:, :3].astype(np.float32)
        
        data = {
            "name": frame_id,
            'points': pts_obs,
            'feat': None,
            # 'calib': calib,
            # 'bounding_boxes': label,
            'Tbl':self.Tbl,
            'Twb':Twb,
            'Twl':Twl
        }
        return data
    def gen_datas(self):
        res = []
        for i in range(10):
            x = i
            Twb = PoseInfo().construct_fromyprt([0, 0, 0], [x, 0, 0])
            res.append(self.gen_data(Twb, "frame_{:03}".format(i)))
        return res
    def run(self):
        return

class AppTest(object):
    def __init__(self):
        # construct a dataset by specifying dataset_path
        self.dataset = NuScenes(dataset_path='/home/levin/workspace/data/temp/nuscenes/v1.0-mini')
        return
    def test_mlui(self):
        vis = Visualizer()
        vis.visualize_dataset(self.dataset, 'train', indices=range(10))
        return
    def test_data(self):
        all_split = self.dataset.get_split('train')
        
        start_ind = 81
        for i in np.arange(start_ind, start_ind + 1):
            f0 = all_split.get_data(i)
            f1 = all_split.get_data(i+ 1)
            
            Twb0 = f0['Twb']
            Twb1 = f1['Twb']
            rel_pose = Twb0.I * Twb1
            print("{}-{}: Tb0b1={}".format(i, i+1, rel_pose.get_short_desc()))
            
            
            Tbl0 = f0['Tbl']
            Tbl1 = f1['Tbl']
            
            Twl0 = Twb0 * Tbl0
            Twl1 = Twb1 * Tbl1
            
            Tlol1 = Tbl0.I * Tbl1
            print("Twb0={}".format(Twb0.get_short_desc()))
            print("Tbl={}".format(Tbl0.get_short_desc()))
            print("{}-{}: Tlol1={}".format(i, i+1, Tlol1.get_short_desc()))
            print("")
            
            pc0 = f0['point']
            pc1 = f1['point']
            
            tcloud0 = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            xyz = pc0[:, [0, 1, 2]]
            tcloud0.point["positions"] = Visualizer._make_tcloud_array(xyz,
                                                                      copy=True)
            # tcloud0.transform(Twl0.T)
            
            
            tcloud1 = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
            xyz = pc1[:, [0, 1, 2]]
            tcloud1.point["positions"] = Visualizer._make_tcloud_array(xyz,
                                                                      copy=True)
            # tcloud1.transform(Twl1.T)
            
            
            o3d.visualization.draw_geometries([tcloud0.to_legacy(), tcloud1.to_legacy()],
                                  )
            
        
        return
    def test_sim(self):
        sim = Simulate3DData()
        data = sim.gen_datas()
        
        vis = Visualizer()
        vis.visualize(data)
        
        # start_ind = 0
        # for i in np.arange(start_ind, start_ind + 1):
        #     f0 = data[i]
        #     f1 = data[i+ 1]
        #
        #     Twb0 = f0['Twb']
        #     Twb1 = f1['Twb']
        #     rel_pose = Twb0.I * Twb1
        #     print("{}-{}: Tb0b1={}".format(i, i+1, rel_pose.get_short_desc()))
        #
        #
        #     Tbl0 = f0['Tbl']
        #     Tbl1 = f1['Tbl']
        #
        #     Twl0 = Twb0 * Tbl0
        #     Twl1 = Twb1 * Tbl1
        #
        #     Tlol1 = Tbl0.I * Tbl1
        #     print("Twb0={}".format(Twb0.get_short_desc()))
        #     print("Tbl={}".format(Tbl0.get_short_desc()))
        #     print("{}-{}: Tlol1={}".format(i, i+1, Tlol1.get_short_desc()))
        #     print("")
        #
        #     pc0 = f0['point']
        #     pc1 = f1['point']
        #
        #     tcloud0 = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
        #     xyz = pc0[:, [0, 1, 2]]
        #     tcloud0.point["positions"] = Visualizer._make_tcloud_array(xyz,
        #                                                               copy=True)
        #     # tcloud0.transform(Twl0.T)
        #
        #
        #     tcloud1 = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
        #     xyz = pc1[:, [0, 1, 2]]
        #     tcloud1.point["positions"] = Visualizer._make_tcloud_array(xyz,
        #                                                               copy=True)
        #     # tcloud1.transform(Twl1.T)
        #
        #     orig = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #     o3d.visualization.draw_geometries([tcloud0.to_legacy(), tcloud1.to_legacy()],
        #                           )
        return
    def run(self):
        self.test_sim()
        # self.test_data()
        # self.test_mlui()
        return
    
    
if __name__ == "__main__":   
    obj= AppTest()
    obj.run()


# dataset = SemanticKITTI(dataset_path='/home/levin/workspace/data/temp/semantic_kiti/SemanticKitti')

# get the 'all' split that combines training, validation and test set
# 
#
# # print the attributes of the first datum
# print(all_split.get_attr(0))
#
# # print the shape of the first point cloud
# 

# show the first 100 frames using the visualizer




# data = [{'name': 'my_point_cloud',
#                     'points': f0['point'],
#     }]
# bounding_boxes = f0['bounding_boxes']
# vis.visualize(data,
#                   lut=None,
#                   bounding_boxes=bounding_boxes,
#                   width=1280,
#                   height=768)