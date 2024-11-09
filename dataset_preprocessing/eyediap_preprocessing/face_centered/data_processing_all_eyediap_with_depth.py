import numpy as np
import scipy.io as sio
import cv2 
import os
import sys

from sqlalchemy import false, true
#sys.path.append("../core/")
import data_processing_core as dpc


root = "/data/EYEDIAP/EYEDIAP/Data"
out_root = "/data/Eyediap_3D_face_depth"
scale =False

def ImageProcessing_Diap():
    folders = os.listdir(root)
    folders.sort(key=lambda x:int(x.split("_")[0]))
    
    count_dict = {}
    for i in range(20):
        count_dict[str(i)] = 0

    for folder in folders:
        if "FT" not in folder:

            depth_path = os.path.join(root, folder, "depth.mov")
            video_path = os.path.join(root, folder, "rgb_vga.mov")
            head_path = os.path.join(root, folder, "head_pose.txt")
            anno_path = os.path.join(root, folder, "eye_tracking.txt")
            camparams_rgb_vga_path = os.path.join(root, folder, "rgb_vga_calibration.txt")
            target_path = os.path.join(root, folder, "screen_coordinates.txt")
            camparams_depth_path = os.path.join(root, folder, "depth_calibration.txt")
            
            number = int(folder.split("_")[0])
            count = count_dict[str(number)]
            person = "p" + str(number)

            im_outpath = os.path.join(out_root, "Image", person+"_rgb")
            im_outpath_depth = os.path.join(out_root, "Image", person+"_depth")
            label_outpath = os.path.join(out_root, "Label", f"{person}.label")

            if not os.path.exists(im_outpath):
                os.makedirs(im_outpath)
            if not os.path.exists(os.path.join(out_root, "Label")):
                os.makedirs(os.path.join(out_root, "Label"))
            if not os.path.exists(label_outpath):
                with open(label_outpath, 'w') as outfile:
                    outfile.write("Face Left Right Face_depth Left_depth Right_depth metapath 3DGaze 3DHead 2DGaze 2DHead Rvec Svec GazeOrigin\n")
                    #([save_name_face, save_name_left, save_name_right, save_name_face_depth, save_name_left_depth, save_name_right_depth, save_metapath, save_gaze, save_head, save_gaze2d, save_head2d, save_rvec, save_svec, save_origin]) 

            rgb_frame_output = os.path.join(out_root, "Image", person+"_rgb_original_frame")
            depth_convert_frame_output = os.path.join(out_root, "Image", person+"_depth_convert_frame")

            print(f"Start Processing p{number}: {folder}")
            count_dict[str(number)] = ImageProcessing_PerVideos(depth_path, video_path, head_path,\
                anno_path, camparams_rgb_vga_path, target_path, im_outpath, im_outpath_depth,\
                label_outpath, folder, count, person, camparams_depth_path,\
                rgb_frame_output, depth_convert_frame_output  )

#MyComment:--------------------perspective transfer Matrix----------------------------------------

def calPerspectiveTransferMatrix(eyes_vals,pointsSet):
    four_points_Set =[]
    for i in range(3):#3 different frame
        eye_coor =[]
        center_left = eyes_vals[4*i:4*i+2]
        center_right = eyes_vals[4*i+2:4*i+4]
        eye_coor.append(center_left)
        eye_coor.append(center_right)
        eye_coor_np = np.float32(eye_coor)
        four_points = np.concatenate((eye_coor_np, pointsSet[i][:2,:]), axis=0)
        four_points_Set.append(four_points)

    four_points_in_vga_rgb = four_points_Set[0]
    four_points_depth = four_points_Set[1]
    four_points_in_hd = four_points_Set[2]

    matrix_depth2vga_rgb = cv2.getPerspectiveTransform(four_points_depth, four_points_in_vga_rgb)
    matrix_hd2vga_rgb = cv2.getPerspectiveTransform(four_points_in_hd, four_points_in_vga_rgb)
    return matrix_depth2vga_rgb,matrix_hd2vga_rgb

def depth_align_to_rgb():
    a=1

def project_points(points, calibration, *args):#points_2D = np.int32(project_points(points, calibration))
    """
    Projects a set of points into the cameras coordinate system
    """
    R = calibration['R']
    T = calibration['T']
    intr = calibration['intrinsics']
    # W.r.t. to the camera coordinate system,
    # MyComment:vga CCS to  HCS of every camera
    points_c = np.dot(points, R) - np.dot(R.transpose(), T).reshape(1, 3)

    points_2D = np.dot(points_c, intr.transpose())
    #MyComment: do not understand next line
    points_2D = points_2D[:,:2]/(points_2D[:, 2].reshape(-1,1))#HCS to chengxiang，x，y->u,v
    return points_2D

def project_points(points, calibration, *args):#points_2D = np.int32(project_points(points, calibration))
    """
    Projects a set of points into the cameras coordinate system
    """
    R = calibration['R']
    T = calibration['T']
    intr = calibration['intrinsics']
    # W.r.t. to the camera coordinate system,
    # MyComment:vga CCS to  HCS of every camera
    points_c = np.dot(points, R) - np.dot(R.transpose(), T).reshape(1, 3)

    points_2D = np.dot(points_c, intr.transpose())
    #MyComment: do not understand next line
    points_2D = points_2D[:,:2]/(points_2D[:, 2].reshape(-1,1))#HCS to chengxiang，x，y->u,v
    return points_2D

def get_axis_in_different_frame(cam_info_rgb_vga, cam_info_depth, vals):#vals head rot trans
    size = 0.05
    points = [[0.0, 0.0, 0.0],
              [size, 0.0, 0.0],
              [0.0, size, 0.0],
              [0.0, 0.0, size]]
    points = np.array(points)

    points+= np.array([0.0, 0.0, 0.13]).reshape(1, 3)
    R = np.array(vals[1: 10]).reshape(3, 3)#headrot in vga CCS
    T = np.array(vals[10: 13]).reshape(3, 1)
    points = np.dot(points, R.transpose())+T.reshape(1, 3)
    #pointsSet = []

    pro_points_rgb_vga = project_points(points, cam_info_rgb_vga)
    # points_2D = np.int32(pro_points_rgb_vga)
    points_2D_rgb_vga_float32 = np.float32(pro_points_rgb_vga)
    pro_points_depth = project_points(points,cam_info_depth )
    points_2D_depth_float32 = np.float32(pro_points_depth)

    
    points_neg = [[0.0, 0.0, 0.0],
              [-size, 0.0, 0.0],
              [0.0, -size, 0.0],
              [0.0, 0.0, size]]
    points_neg = np.array(points_neg)
    points_neg+= np.array([0.0, 0.0, 0.13]).reshape(1, 3)
    points_neg = np.dot(points_neg, R.transpose())+T.reshape(1, 3)
    pro_points_rgb_vga_neg = project_points(points_neg, cam_info_rgb_vga)
    # points_2D = np.int32(pro_points_rgb_vga)
    points_2D_rgb_vga_float32_neg = np.float32(pro_points_rgb_vga_neg)
    pro_points_depth_neg = project_points(points_neg,cam_info_depth )
    points_2D_depth_float32_neg = np.float32(pro_points_depth_neg)




    return points_2D_rgb_vga_float32, points_2D_depth_float32, points_2D_rgb_vga_float32_neg, points_2D_depth_float32_neg

def draw_point(frame, point, size= 5, thickness=2):
    """
    Draw a point in the given image as a yellow "+"
    """
    point = int(point[0]), int(point[1])
    cv2.line(frame, (point[0]-size, point[1]), (point[0]+size, point[1]), (0, 255, 255), lineType=cv2.LINE_AA, thickness =thickness)
    cv2.line(frame, (point[0]  , point[1]-size), (point[0], point[1]+size), (0, 255, 255), thickness =thickness)

def draw_line(frame, point_1, point_2, size=5, color=(0,255,255), thickness =2):
    cv2.line(frame, (point_1[0], point_1[1]), (point_2[0], point_2[1]), color, lineType=cv2.LINE_AA, thickness =thickness)

def ImageProcessing_PerVideos(depth_path, video_path, head_path, \
    anno_path, camparams_rgb_vga_path, target_path, im_outpath, \
    im_outpath_depth,label_outpath, folder, count, person,\
    camparams_depth_path, rgb_frame_output, depth_convert_frame_output  ):


#   ImageProcessing_PerVideos(video_path, head_path,anno_path, camparams_path, target_path, im_outpath, label_outpath, folder, count, person)
# video_path = rgb_vga.mov   
# head_path = head_pose.txt
# anno_path = eye_tracking.txt eye in every camera, 3D position
# camparams_path = rgb_vga_calibration.txt
# target_path = screen_coordinates.txt

    # Read annotations
    with open(head_path) as infile:
        head_info = infile.readlines()
    with open(anno_path) as infile:
        anno_info = infile.readlines()
    with open(target_path) as infile:
        target_info = infile.readlines()
    length = len(target_info) - 1 # remove first line ,table head

    # Read camera parameters
    cam_info_rgb_vga = CamParamsDecode(camparams_rgb_vga_path)
    camera_rgb_vga = cam_info_rgb_vga["intrinsics"]
    cam_rot_rgb_vga = cam_info_rgb_vga["R"]
    cam_trans_rgb_vga = cam_info_rgb_vga["T"]*1000 #MyComment:convert m->mm

    cam_info_depth = CamParamsDecode(camparams_depth_path)
    camera_depth = cam_info_depth["intrinsics"]
    cam_rot_depth= cam_info_depth["R"]
    cam_trans_depth = cam_info_depth["T"]*1000 #MyComment:convert m->mm   


    # Read video
    cap = cv2.VideoCapture(video_path)
    cap_depth = cv2.VideoCapture(depth_path)

    # create handle of label
    outfile = open(label_outpath, 'a')
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))

    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))
    

    if not os.path.exists(os.path.join(im_outpath_depth, "left_depth")):
        os.makedirs(os.path.join(im_outpath_depth, "left_depth"))
    
    if not os.path.exists(os.path.join(im_outpath_depth, "right_depth")):
        os.makedirs(os.path.join(im_outpath_depth, "right_depth"))

    if not os.path.exists(os.path.join(im_outpath_depth, "face_depth")):
        os.makedirs(os.path.join(im_outpath_depth, "face_depth"))   

    if not os.path.exists(rgb_frame_output):
        os.makedirs(rgb_frame_output) 

    if not os.path.exists(depth_convert_frame_output):
        os.makedirs(depth_convert_frame_output)      

    num = 1
    # Image Processing 
    got_matrix_flag = false
   # length =  300
    for index in range(1, length+1):
        ret, frame = cap.read()
        ret_depth, depth_frame = cap_depth.read()
        if (index-1) % 15 != 0:
            continue

        size = frame.shape
        w = size[1] 
        h = size[0] 

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(index/length * 20))
        progressbar = "\r" + progressbar + f" {index}|{length}"
        print(progressbar, end="", flush=True)

        # Calculate rotation and transition of head pose.
        head = head_info[index]
        head = list(map(eval, head.strip().split(";")))
        if len(head) != 13:
            print("[Error Head]")
            continue
        
        head_rot = head[1:10]
        head_rot = np.array(head_rot).reshape([3,3])
        head1 = cv2.Rodrigues(head_rot)[0].T[0]
        head2d = dpc.HeadTo2d(head1)
        print(head2d)
        print("rongyu:------")

        head_rot = np.dot(cam_rot_rgb_vga, head_rot) 
        head1 = cv2.Rodrigues(head_rot)[0].T[0]
        head2d = dpc.HeadTo2d(head1)
        print(head2d)
        print("------")
#        exit()

            # rotate the head into camera coordinate system
        head_trans = np.array(head[10:13])*1000#MyComment:m->mm
        head_trans = np.dot(cam_rot_rgb_vga, head_trans)
        head_trans = head_trans + cam_trans_rgb_vga#MyComment:HCS->WCS

        # Calculate the 3d coordinates of origin.
        anno = anno_info[index]
        anno = list(map(eval, anno.strip().split(";")))
        if len(anno) != 19:
            print("[Error anno]")
            continue
        anno = np.array(anno)

        left3d = anno[13:16]*1000
        left3d = np.dot(cam_rot_rgb_vga, left3d) + cam_trans_rgb_vga #MyComment:CCS->WCS point
        right3d = anno[16:19]*1000
        right3d = np.dot(cam_rot_rgb_vga, right3d) + cam_trans_rgb_vga #MyComment:CCS->WCS

        face3d = (left3d + right3d)/2
        face3d = (face3d + head_trans)/2#MyComment:WCS

        left2d = anno[1:3]#MyComment: Kinect RGB eyeball center left and right (x_l,y_l,x_r,y_r)
        right2d= anno[3:5]

        left2d_depth = anno[5:7]
        right2d_depth = anno[7:9] #MyComment: pick_four_points in rgb and depth frame resspectively


        #if(got_matrix_flag==false):

        points_2D_rgb_vga_4inHCCS_Axis, points_2D_depth_4inHCCS_Axis,points_2D_rgb_vga_float32_neg, points_2D_depth_float32_neg =  get_axis_in_different_frame(cam_info_rgb_vga, cam_info_depth,head)
        
        eye_coor_rgb_vga = []
        # eye_coor_rgb_vga.append(left2d)
        # eye_coor_rgb_vga.append(right2d)
        # eye_coor_rgb_vga_np = np.float32(eye_coor_rgb_vga)
        # four_points_rgb_vga = np.concatenate((eye_coor_rgb_vga_np, points_2D_rgb_vga_4inHCCS_Axis[:2,:]), axis=0)
        four_points_rgb_vga = np.concatenate((points_2D_rgb_vga_float32_neg[1:3,:], points_2D_rgb_vga_4inHCCS_Axis[1:3,:]), axis=0)

        eye_coor_depth = []
        # eye_coor_depth.append(left2d_depth)
        # eye_coor_depth.append(right2d_depth)
        # eye_coor_depth_np = np.float32(eye_coor_depth)
        # four_points_depth = np.concatenate((eye_coor_depth_np, points_2D_depth_4inHCCS_Axis[:2,:]), axis=0)       
        four_points_depth = np.concatenate((points_2D_depth_float32_neg[1:3,:], points_2D_depth_4inHCCS_Axis[1:3,:]), axis=0)  
        
        
        matrix_depth2vga_rgb = cv2.getPerspectiveTransform(four_points_depth, four_points_rgb_vga)
        got_matrix_flag = true

        depth_aligned = cv2.warpPerspective(depth_frame, matrix_depth2vga_rgb, (w,h))  

    

        # Calculate the 3d coordinates of target
        target = target_info[index] #MyComment:screen_coordinates.txt
        target = list(map(eval,target.strip().split(";")))
        if len(target) != 6:
            print("[Error target]")
            continue
        target3d = np.array(target)[3:6]*1000 #3D coordinates (x,y,z)

        # target3d = target3d.T - cam_trans
        # target3d = np.dot(np.linalg.inv(cam_rot), target3d)

        # Normalize the left eye image
        norm = dpc.norm(center = face3d,
                        gazetarget = target3d,
                        headrotvec = head_rot,
                        imsize = (224, 224),
                        camparams = camera_rgb_vga)

        # Acquire essential info
        im_face = norm.GetImage(frame)
        im_depth_face = norm.GetImage_depth(depth_aligned)

        gaze = norm.GetGaze(scale=scale)
        head = norm.GetHeadRot(vector=False)
        head = cv2.Rodrigues(head)[0].T[0]

        origin = norm.GetCoordinate(face3d)#MyComment
            #====MyComment:locate origin in face 
        origin2d = norm.convert3dorigin_2_face(origin)
        #draw_point(im_face, origin2d, size= 5, thickness=2)
    
        imshowpath  = "/home/workspace/feifeizhang/GAZE/show_eye_in_crop_face_"+"person"+str(person)
        if(not os.path.exists(imshowpath)):
            os.mkdir(imshowpath)

        #cv2.imwrite(os.path.join( imshowpath,str(index)+".png"), im_face)
        

        rvec, svec = norm.GetParams() # virtual camera parameter
        gaze2d = dpc.GazeTo2d(gaze)
        head2d = dpc.HeadTo2d(head)

        # Crop Eye Image
        left2d = norm.GetNewPos(left2d)
        right2d = norm.GetNewPos(right2d)

        #======MyComment ~quick check location====== 
        #Crop Eye Image
        # draw_point(im_face, left2d, size= 5, thickness=2)
        # draw_point(im_face, right2d, size= 5, thickness=2)
        # imshowpath  = "/home/workspace/feifeizhang/GAZE/show_eye_in_crop_face_"+"person"+str(person)
        # if(not os.path.exists(imshowpath)):
        #     os.mkdir(imshowpath)

        # cv2.imwrite(os.path.join( imshowpath,str(index)+".png"), im_face)

        #   im_face = ipt.circle(im_face, left2d, 2)
        #im_face = ipt.circle(im_face, [left2d[0]+10, left2d[1]], 2)
        # cv2.imwrite("eye.jpg", im_face)
        
        
        im_left = norm.CropEyeWithCenter(left2d)
    #    im_left = dpc.EqualizeHist(im_left)
        im_right = norm.CropEyeWithCenter(right2d)
    #    im_right = dpc.EqualizeHist(im_right)
        im_depth_left = norm.CropEyeWithCenter_depth(left2d)
        im_depth_right = norm.CropEyeWithCenter_depth(right2d)
        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, "face", str(count + num)+".jpg"), im_face)
        cv2.imwrite(os.path.join(im_outpath, "left", str(count + num)+".jpg"), im_left)
        cv2.imwrite(os.path.join(im_outpath, "right", str(count + num)+".jpg"), im_right)

        cv2.imwrite(os.path.join(im_outpath_depth, "face_depth", str(count + num)+".jpg"), im_depth_face)
        cv2.imwrite(os.path.join(im_outpath_depth, "left_depth", str(count + num)+".jpg"), im_depth_left)
        cv2.imwrite(os.path.join(im_outpath_depth, "right_depth", str(count + num)+".jpg"), im_depth_right)

        cv2.imwrite(os.path.join(rgb_frame_output, "original_rgb" + str(count + num)+".jpg"), frame)
        cv2.imwrite(os.path.join(depth_convert_frame_output, "depth_convert" + str(count + num)+".jpg"), depth_aligned)
        
        save_name_face = os.path.join(person + "_rgb" , "face", str(count + num) + ".jpg")
        save_name_left = os.path.join(person + "_rgb" , "left", str(count + num) + ".jpg")
        save_name_right = os.path.join(person + "_rgb" , "right", str(count + num) + ".jpg")

        save_name_face_depth = os.path.join(person + "_depth", "face_depth", str(count + num) + ".jpg")
        save_name_left_depth = os.path.join(person + "_depth", "left_depth", str(count + num) + ".jpg")
        save_name_right_depth = os.path.join(person + "_depth", "right_depth", str(count + num) + ".jpg")

        save_metapath = folder + f"_{index}"
        # save_flag = "left"
        save_gaze = ",".join(gaze.astype("str"))
        save_head = ",".join(head.astype("str"))
        save_gaze2d = ",".join(gaze2d.astype("str"))
        save_head2d = ",".join(head2d.astype("str"))
        save_origin = ",".join(origin.astype("str"))
        save_rvec = ",".join(rvec.astype("str"))
        save_svec = ",".join(svec.astype("str"))

        save_str = " ".join([save_name_face, save_name_left, save_name_right, save_name_face_depth, save_name_left_depth, save_name_right_depth, save_metapath, save_gaze, save_head, save_gaze2d, save_head2d, save_rvec, save_svec, save_origin]) 
        outfile.write(save_str + "\n")
        num += 1

    count += (num-1)
    outfile.close()
    print("")
    return count

def CamParamsDecode(path):
    cal = {}
    fh = open(path, 'r')
    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(';')]
    cal['size'] = cal['size'][0], cal['size'][1]
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['intrinsics'] = np.array(vals).reshape(3, 3)
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['R'] = np.array(vals).reshape(3, 3)
    # Read the [T] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['T'] = np.array(vals).reshape(3)
    fh.close()
    return cal


if __name__ == "__main__":
    ImageProcessing_Diap()
