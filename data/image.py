import os
import argparse
import warnings
from tqdm import tqdm
import pickle
#--------------------------------------------------------------#
#-------------------------获取AGQA的图片image--------------------#
#--------------------------------------------------------------#
train_sgg_file=pickle.load(open('/home/lll/AGQA-dest/dest/data/AGQA_scene_graphs/AGQA_train_stsgs.pkl','rb'))
test_sgg_file=pickle.load(open('/home/lll/AGQA-dest/dest/data/AGQA_scene_graphs/AGQA_test_stsgs.pkl','rb'))

train=train_sgg_file.keys()
test=test_sgg_file.keys()

file_dir='/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/FRCNN/FRNN_train_Frame_imgs'
files=os.listdir(file_dir)
import shutil
pbar = tqdm(total=len(files))

for file in files:
    mp4=file.replace('.mp4','')
    file_path=os.path.join(file_dir,file)
    if mp4 in train:
        move_path=os.path.join('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Image/train',file)
        shutil.copytree(file_path, move_path)
    elif mp4 in test:
        move_path=os.path.join('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Image/test',file)
        shutil.copytree(file_path, move_path)
    pbar.update(1)
pbar.close()

'''
train=['00HFP','00MFE','00N38']
test=['00T1E','0A8CF']'''
'''def dump_frames(args):
    video_dir = args.video_dir
    frame_train_dir = args.frame_train_dir
    frame_test_dir = args.frame_test_dir
    annotation = args.annotation
    all_frames = False

    # Load the list of annotated frames
    frame_list = []
    with open(annotation, 'r') as f:
        for frame in f:
            frame_list.append(frame.rstrip('\n'))

    # Create video to frames mapping
    video2frames = {}
    for path in frame_list:
        video, frame = path.split('/')
        if video not in video2frames:
            video2frames[video] = []
        video2frames[video].append(frame)

    # For each video, dump frames.
    for v in tqdm(video2frames):
        mp4=v.replace('.mp4','')
        if mp4 in train:
            curr_frame_dir = os.path.join(frame_train_dir, v)
        elif mp4 in test:
            curr_frame_dir = os.path.join(frame_test_dir, v)

        if not os.path.exists(curr_frame_dir):
            os.makedirs(curr_frame_dir)
            # Use ffmpeg to extract frames. Different versions of ffmpeg may generate slightly different frames.
            # We used ffmpeg 2.8.15 to dump our frames.
            # Note that the frames are extracted according to their original video FPS, which is not always 24.
            # Therefore, our frame indices are different from Charades extracted frames' indices.
            os.system('ffmpeg -loglevel panic -i %s/%s %s/%%06d.png' % (video_dir, v, curr_frame_dir))

            # if not keeping all frames, only keep the annotated frames included in frame_list.txt
            if not all_frames:
                keep_frames = video2frames[v]
                frames_to_delete = set(os.listdir(curr_frame_dir)) - set(keep_frames)
                for frame in frames_to_delete:
                    os.remove(os.path.join(curr_frame_dir, frame))
        else:
            warnings.warn('Frame directory %s already exists. Skipping dumping into this directory.' % curr_frame_dir,
                          RuntimeWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump frames")
    parser.add_argument("--video_dir", default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/Video/",
                        help="Folder containing Charades videos.")
    parser.add_argument("--frame_train_dir", default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Image/train",
                        help="Root folder containing frames to be dumped.")
    parser.add_argument("--frame_test_dir", default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Image/test",
                        help="Root folder containing frames to be dumped.")
    parser.add_argument("--annotation", default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/frame_list.txt",
                        help=("Folder containing annotation files---frame_list.txt."))
    #parser.add_argument("--all_frames", action="store_true",
    #                    help="Set if you want to dump all frames, rather than the frames listed in frame_list.txt")
    args = parser.parse_args()
    dump_frames(args)'''



