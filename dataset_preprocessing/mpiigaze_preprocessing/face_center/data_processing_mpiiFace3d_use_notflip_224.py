import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
sys.path.append("../core/")
import data_processing_core_not_get_lmk as dpc

root = "/data/MPIIFaceGaze"
               
sample_root = "/data/MPIIGaze/Evaluation Subset/sample list for eye image"
out_root = "/data/Facecenter_MPIIFaceGaze_224_not_flip"
scale = True

def ImageProcessing_MPII():
    persons = os.listdir(sample_root)
    persons.sort()
    for person in persons:
        sample_list = os.path.join(sample_root, person) 

        person = person.split(".")[0]
        im_root = os.path.join(root, person)
        anno_path = os.path.join(root, person, f"{person}.txt")

        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "Label", f"{person}.label")

        if not os.path.exists(im_outpath):
            os.makedirs(im_outpath)
        if not os.path.exists(os.path.join(out_root, "Label")):
            os.makedirs(os.path.join(out_root, "Label"))

        print(f"Start Processing {person}")
        ImageProcessing_Person(im_root, anno_path, sample_list, im_outpath, label_outpath, person)


def ImageProcessing_Person(im_root, anno_path, sample_list, im_outpath, label_outpath, person):
    # Read camera matrix
    camera = sio.loadmat(os.path.join(f"{im_root}", "Calibration", "Camera.mat"))
    camera = camera["cameraMatrix"]

    # Read gaze annotation
    annotation = os.path.join(anno_path)
    with open(annotation) as infile:
        anno_info = infile.readlines()
    anno_dict = {line.split(" ")[0]: line.strip().split(" ")[1:-1] for line in anno_info}

    # Create the handle of label 
    outfile = open(label_outpath, 'w')
    outfile.write("Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin\n")
    if not os.path.exists(os.path.join(im_outpath, "face")):
        os.makedirs(os.path.join(im_outpath, "face"))
    if not os.path.exists(os.path.join(im_outpath, "left")):
        os.makedirs(os.path.join(im_outpath, "left"))
    if not os.path.exists(os.path.join(im_outpath, "right")):
        os.makedirs(os.path.join(im_outpath, "right"))

    # Image Processing 
    with open(sample_list) as infile:
        im_list = infile.readlines()
        total = len(im_list)

    for count, info in enumerate(im_list):

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/total * 20))
        progressbar = "\r" + progressbar + f" {count}|{total}"
        print(progressbar, end = "", flush=True)

        # Read image info
        im_info, which_eye = info.strip().split(" ")
        day, im_name = im_info.split("/")
        im_number = int(im_name.split(".")[0])        

        # Read image annotation and image
        im_path = os.path.join(im_root, day, im_name)
        im = cv2.imread(im_path)
        annotation = anno_dict[im_info]
        annotation = AnnoDecode(annotation) 
        origin = annotation["facecenter"]

        # Normalize the image
        norm = dpc.norm(center = annotation["facecenter"],
                        gazetarget = annotation["target"],
                        headrotvec = annotation["headrotvectors"],
                        imsize = (224, 224),
                        camparams = camera)

        im_face = norm.GetImage(im)

        # Crop left eye images
        llc = norm.GetNewPos(annotation["left_left_corner"])
        lrc = norm.GetNewPos(annotation["left_right_corner"])
        im_left = norm.CropEye(llc, lrc)
        #im_left = dpc.EqualizeHist(im_left)
        
        # Crop Right eye images
        rlc = norm.GetNewPos(annotation["right_left_corner"])
        rrc = norm.GetNewPos(annotation["right_right_corner"])
        im_right = norm.CropEye(rlc, rrc)
        #im_right = dpc.EqualizeHist(im_right)
 
        # Acquire essential info
        gaze = norm.GetGaze(scale=scale)
        head = norm.GetHeadRot(vector=True)
        origin = norm.GetCoordinate(annotation["facecenter"])
        rvec, svec = norm.GetParams()

        # flip the images when it is right eyes


        gaze_2d = dpc.GazeTo2d(gaze)
        head_2d = dpc.HeadTo2d(head)
   
        # Save the acquired info
        cv2.imwrite(os.path.join(im_outpath, "face", str(count+1)+".jpg"), im_face)
        cv2.imwrite(os.path.join(im_outpath, "left", str(count+1)+".jpg"), im_left)
        cv2.imwrite(os.path.join(im_outpath, "right", str(count+1)+".jpg"), im_right)
        
        save_name_face = os.path.join(person, "face", str(count+1) + ".jpg")
        save_name_left = os.path.join(person, "left", str(count+1) + ".jpg")
        save_name_right = os.path.join(person, "right", str(count+1) + ".jpg")

        save_origin = im_info
        save_flag = which_eye
        save_gaze = ",".join(gaze.astype("str"))
        save_head = ",".join(head.astype("str"))
        save_gaze2d = ",".join(gaze_2d.astype("str"))
        save_head2d = ",".join(head_2d.astype("str"))
        save_rvec = ",".join(rvec.astype("str"))
        save_svec = ",".join(svec.astype("str"))
        origin = ",".join(origin.astype("str"))

        save_str = " ".join([save_name_face, save_name_left, save_name_right, save_origin, save_flag, save_gaze, save_head, save_gaze2d, save_head2d, save_rvec, save_svec, origin])
        
        outfile.write(save_str + "\n")
    print("")
    outfile.close()

def AnnoDecode(anno_info):
	annotation = np.array(anno_info).astype("float32")
	out = {}
	out["left_left_corner"] = annotation[2:4]
	out["left_right_corner"] = annotation[4:6]
	out["right_left_corner"] = annotation[6:8]
	out["right_right_corner"] = annotation[8:10]
	out["headrotvectors"] = annotation[14:17]
	out["headtransvectors"] = annotation[17:20]
	out["facecenter"] = annotation[20:23]
	out["target"] = annotation[23:26]
	return out


if __name__ == "__main__":
    ImageProcessing_MPII()
