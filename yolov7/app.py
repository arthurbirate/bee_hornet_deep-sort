import streamlit as st
import mimetypes
import argparse
import time
from pathlib import Path
import time
import cv2
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *

import streamlit as st
import mimetypes
import argparse
import time
from pathlib import Path
import time
import cv2
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
from numpy import random


from models.experimental import attempt_download, attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *


st.set_page_config(layout="centered")
video_processed = False
st.title('BeeSafe')
st.subheader(
    'BeeSafe: Revolutionizing Bee and Hornet Monitoring with YOLO-Driven Tracking')
st.markdown('Explore the cutting-edge world of bee and hornet surveillance with BeeSafe! This project leverages YOLO (You Only Look Once) technology to track and monitor these vital pollinators in real-time. Dive into the buzzworthy intersection of technology and environmental conservation. 🐝🔍 #BeeSafe #YOLOTracking #Environmental Innovation')

st.header("Try it out 🚀")
tab1, tab2, tab3 = st.tabs(["A video", '             ', "An image"])


with tab1:

    st.markdown("<br>", unsafe_allow_html=True)

    video_bytes = None
    video = st.file_uploader("Upload a Video", type=[
        "mp4", "mpeg"])

    st.markdown("<br>", unsafe_allow_html=True)

    options_mapping = {'Hornet': 1, 'Bee': 0}

    optionss = st.multiselect(
        key='my_video_key',
        label='Choose to track hornet or both bee and hornet',
        options=list(options_mapping.keys())
    )

    selected_values = [options_mapping[option] for option in optionss]

    st.markdown("<br>", unsafe_allow_html=True)

    thickness = st.slider("Choose thickness of the box", 0, 5)

    st.markdown("<br>", unsafe_allow_html=True)

    processed_video_container = st.empty()

    process_video_button = st.button("Process Video")
    if process_video_button:
        with st.spinner("Processing..."):
            if video is not None and thickness is not None and thickness != 0:

                def draw_boxes_video(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
                    for i, box in enumerate(bbox):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        tl = opt.thickness or round(
                            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

                        cat = int(
                            categories[i]) if categories is not None else 0
                        id = int(identities[i]
                                 ) if identities is not None else 0
                        # conf = confidences[i] if confidences is not None else 0

                        color = colors[cat]

                        if not opt.nobbox:
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

                        if not opt.nolabel:
                            label = str(
                                id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(
                                label, 0, fontScale=tl / 3, thickness=tf)[0]
                            c2 = x1 + t_size[0], y1 - t_size[1] - 3
                            cv2.rectangle(img, (x1, y1), c2, color, -
                                          1, cv2.LINE_AA)  # filled
                            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3,
                                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                    return img

                def detect_video(save_img=True):
                    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
                    save_img = not opt.nosave and not source.endswith(
                        '.txt')  # save inference images
                    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                        ('rtsp://', 'rtmp://', 'http://', 'https://'))
                    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                    exist_ok=opt.exist_ok))  # increment run
                    if not opt.nosave:
                        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                                              exist_ok=True)  # make dir

                    # Initialize
                    set_logging()
                    device = select_device(opt.device)
                    half = device.type != 'cpu'  # half precision only supported on CUDA

                    # Load model
                    model = attempt_load(
                        weights, map_location=device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size

                    if trace:
                        model = TracedModel(model, device, opt.img_size)

                    if half:
                        model.half()  # to FP16

                    # Second-stage, classifier
                    classify = False
                    if classify:
                        modelc = load_classifier(
                            name='resnet101', n=2)  # initialize
                        modelc.load_state_dict(torch.load(
                            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

                    # Set Dataloader
                    vid_path, vid_writer = None, None
                    if webcam:
                        view_img = check_imshow()
                        cudnn.benchmark = True  # set True to speed up constant image size inference
                        dataset = LoadStreams(
                            source, img_size=imgsz, stride=stride)
                    else:
                        dataset = LoadImages(
                            source, img_size=imgsz, stride=stride)

                    # Get names and colors
                    names = model.module.names if hasattr(
                        model, 'module') else model.names
                    colors = [[random.randint(0, 255)
                               for _ in range(3)] for _ in names]

                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                            next(model.parameters())))  # run once
                    old_img_w = old_img_h = imgsz
                    old_img_b = 1

                    t0 = time.time()
                    ###################################
                    startTime = 0
                    ###################################
                    for path, img, im0s, vid_cap in dataset:
                        img = torch.from_numpy(img).to(device)
                        img = img.half() if half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        # Warmup
                        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                            old_img_b = img.shape[0]
                            old_img_h = img.shape[2]
                            old_img_w = img.shape[3]
                            for i in range(3):
                                model(img, augment=opt.augment)[0]

                        # Inference
                        t1 = time_synchronized()
                        pred = model(img, augment=opt.augment)[0]
                        t2 = time_synchronized()

                        # Apply NMS
                        pred = non_max_suppression(
                            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                        t3 = time_synchronized()

                        # Apply Classifier
                        if classify:
                            pred = apply_classifier(pred, modelc, img, im0s)

                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            if webcam:  # batch_size >= 1
                                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                                ), dataset.count
                            else:
                                p, s, im0, frame = path, '', im0s, getattr(
                                    dataset, 'frame', 0)

                            p = Path(p)  # to Path
                            save_path = str(save_dir / p.name)  # img.jpg
                            txt_path = str(save_dir / 'labels' / p.stem) + \
                                ('' if dataset.mode ==
                                 'image' else f'_{frame}')  # img.txt
                            # normalization gain whwh
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(
                                    img.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, -1].unique():
                                    # detections per class
                                    n = (det[:, -1] == c).sum()
                                    # add to string
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                                dets_to_sort = np.empty((0, 6))
                                # NOTE: We send in detected object class too
                                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                    dets_to_sort = np.vstack((dets_to_sort,
                                                              np.array([x1, y1, x2, y2, conf, detclass])))

                                if opt.track:

                                    tracked_dets = sort_tracker.update(
                                        dets_to_sort, opt.unique_track_color)
                                    tracks = sort_tracker.getTrackers()

                                    # draw boxes for visualization
                                    if len(tracked_dets) > 0:
                                        bbox_xyxy = tracked_dets[:, :4]
                                        identities = tracked_dets[:, 8]
                                        categories = tracked_dets[:, 4]
                                        confidences = None

                                        if opt.show_track:
                                            # loop over tracks
                                            for t, track in enumerate(tracks):

                                                track_color = colors[int(
                                                    track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                                # Extracting the centroids for the track
                                                centroids = [(int(coord[0]), int(coord[1]))
                                                             for coord in track.centroidarr]

                                                # Draw lines connecting centroids
                                                for i in range(len(centroids) - 1):
                                                    cv2.line(
                                                        im0, centroids[i], centroids[i + 1], track_color, thickness=opt.thickness)

                                    bbox_xyxy = dets_to_sort[:, :4]
                                    identities = None
                                    categories = dets_to_sort[:, 5]
                                    confidences = dets_to_sort[:, 4]

                                im0 = draw_boxes_video(im0, bbox_xyxy, identities,
                                                       categories, confidences, names, colors)
                            if dataset.mode != 'image' and opt.show_fps:
                                currentTime = time.time()

                                fps = 1/(currentTime - startTime)
                                startTime = currentTime
                                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                            # Save results (image with detections)
                            if save_img:
                                if dataset.mode == 'image':
                                    cv2.imwrite(save_path, im0)
                                    print(
                                        f" The image with the result is saved in: {save_path}")
                                else:  # 'video' or 'stream'
                                    if vid_path != save_path:  # new video
                                        vid_path = save_path
                                        if isinstance(vid_writer, cv2.VideoWriter):
                                            vid_writer.release()  # release previous video writer
                                        if vid_cap:  # video
                                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                            w = int(vid_cap.get(
                                                cv2.CAP_PROP_FRAME_WIDTH))
                                            h = int(vid_cap.get(
                                                cv2.CAP_PROP_FRAME_HEIGHT))
                                        else:  # stream
                                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                                            save_path += '.mp4'
                                        vid_writer = cv2.VideoWriter(
                                            save_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
                                    vid_writer.write(im0)

                                    processed_video_container.image(
                                        im0, channels="BGR", use_column_width=True)

                    video_bytes = open(save_path, 'rb').read()

                    processed_video_container.video(video_bytes)

                    if save_txt or save_img:
                        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                        # print(f"Results saved to {save_dir}{s}")

                    # print(f'Done. ({time.time() - t0:.3f}s)')

                if __name__ == '__main__':
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--weights', nargs='+', type=str,
                                        default='yolov7.pt', help='model.pt path(s)')
                    # file/folder, 0 for webcam
                    parser.add_argument('--source', type=str,
                                        default='inference/images', help='source')
                    parser.add_argument('--img-size', type=int, default=640,
                                        help='inference size (pixels)')
                    parser.add_argument('--conf-thres', type=float,
                                        default=0.25, help='object confidence threshold')
                    parser.add_argument('--iou-thres', type=float,
                                        default=0.45, help='IOU threshold for NMS')
                    parser.add_argument('--device', default='',
                                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                    parser.add_argument('--view-img', action='store_true',
                                        help='display results')
                    parser.add_argument('--save-txt', action='store_true',
                                        help='save results to *.txt')
                    parser.add_argument('--save-conf', action='store_true',
                                        help='save confidences in --save-txt labels')
                    parser.add_argument('--nosave', action='store_true',
                                        help='do not save images/videos')
                    parser.add_argument('--classes', nargs='+', type=int,
                                        help='filter by class: --class 0, or --class 0 2 3')
                    parser.add_argument('--agnostic-nms', action='store_true',
                                        help='class-agnostic NMS')
                    parser.add_argument('--augment', action='store_true',
                                        help='augmented inference')
                    parser.add_argument('--update', action='store_true',
                                        help='update all models')
                    parser.add_argument('--project', default='runs/detect',
                                        help='save results to project/name')
                    parser.add_argument('--name', default='exp',
                                        help='save results to project/name')
                    parser.add_argument('--exist-ok', action='store_true',
                                        help='existing project/name ok, do not increment')
                    parser.add_argument('--no-trace', action='store_true',
                                        help='don`t trace model')

                    parser.add_argument('--track', action='store_true',
                                        help='run tracking')
                    parser.add_argument('--show-track', action='store_true',
                                        help='show tracked path')
                    parser.add_argument(
                        '--show-fps', action='store_true', help='show fps')
                    parser.add_argument('--thickness', type=int, default=2,
                                        help='bounding box and font size thickness')
                    parser.add_argument('--seed', type=int, default=1,
                                        help='random seed to control bbox colors')
                    parser.add_argument('--nobbox', action='store_true',
                                        help='don`t show bounding box')
                    parser.add_argument('--nolabel', action='store_true',
                                        help='don`t show label')
                    parser.add_argument('--unique-track-color', action='store_true',
                                        help='show each track in unique color')

                    with open("temp_video.mp4", "wb") as f:
                        f.write(video.read())

                    opt = argparse.Namespace(
                        weights='best_2.pt',
                        source="temp_video.mp4",
                        no_trace=True,
                        view_img=True,
                        nosave=False,
                        track=True,
                        classes=selected_values,
                        show_track=True,
                        img_size=432,
                        seed=1,
                        update=False,
                        save_txt=False,
                        project='runs/detect',  # Add the project attribute here
                        name='exp',
                        exist_ok=True,
                        device="cpu",
                        augment=False,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        agnostic_nms=False,
                        show_fps=False,
                        unique_track_color=False,
                        thickness=thickness,
                        nobbox=False,
                        nolabel=False
                    )

                    # print(opt)

                # Set random seed
                    np.random.seed(opt.seed)

                    # Initialize Sort tracker
                    sort_tracker = Sort(max_age=5,
                                        min_hits=2,
                                        iou_threshold=0.2)

                    # check_requirements(exclude=('pycocotools', 'thop'))

                    with torch.no_grad():
                        detect_video()

                # Open video file and read its bytes

                    # Display the MIME type
            else:
                st.info(
                    "Please upload a video and choose and choose thickness for bounding boxes.", icon="ℹ️")
        video_file_path = 'runs/detect/exp/temp_video.mp4'
        video_file = open(video_file_path, 'rb')
        video_bytes = video_file.read()
        with st.container():  # Adjust width as needed
            # Display the video
            st.video(video_bytes)

        with open(video_file_path, "rb") as file:
            btn = st.download_button(
                label="Download Video",
                data=file,
                file_name="temp_video.mp4",
                mime="video/mp4"
            )


with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    image_bytes = None
    image = st.file_uploader("Upload an image", type=[
        "jpeg", "webp", "png"], key="tab3")

    st.markdown("<br>", unsafe_allow_html=True)

    options_mapping = {'Hornet': 1, 'Bee': 0}

    options = st.multiselect(
        'Choose to track hornet or both bee and hornet',
        list(options_mapping.keys())
    )

    selected_values = [options_mapping[option] for option in options]

    st.markdown("<br>", unsafe_allow_html=True)

    thickness_image = st.slider(
        "Choose thickness of the box", 0, 4, key="tab3_1")

    st.markdown("<br>", unsafe_allow_html=True)

    processed_image_container = st.empty()

    process_image_button = st.button("Process Image", key="process")
    if process_image_button:
        with st.spinner("Processing..."):
            if image is not None and thickness_image is not None and options is not None and thickness_image != 0:

                def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
                    for i, box in enumerate(bbox):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        tl = opt.thickness or round(
                            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

                        cat = int(
                            categories[i]) if categories is not None else 0
                        id = int(identities[i]
                                 ) if identities is not None else 0
                        # conf = confidences[i] if confidences is not None else 0

                        color = colors[cat]

                        if not opt.nobbox:
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

                        if not opt.nolabel:
                            label = str(
                                id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(
                                label, 0, fontScale=tl / 3, thickness=tf)[0]
                            c2 = x1 + t_size[0], y1 - t_size[1] - 3
                            cv2.rectangle(img, (x1, y1), c2, color, -
                                          1, cv2.LINE_AA)  # filled
                            cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3,
                                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                    return img

                def detect(save_img=True):
                    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
                    save_img = not opt.nosave and not source.endswith(
                        '.txt')  # save inference images
                    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                        ('rtsp://', 'rtmp://', 'http://', 'https://'))
                    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                    exist_ok=opt.exist_ok))  # increment run
                    if not opt.nosave:
                        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                                              exist_ok=True)  # make dir

                    # Initialize
                    set_logging()
                    device = select_device(opt.device)
                    half = device.type != 'cpu'  # half precision only supported on CUDA

                    # Load model
                    model = attempt_load(
                        weights, map_location=device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check img_size

                    if trace:
                        model = TracedModel(model, device, opt.img_size)

                    if half:
                        model.half()  # to FP16

                    # Second-stage, classifier
                    classify = False
                    if classify:
                        modelc = load_classifier(
                            name='resnet101', n=2)  # initialize
                        modelc.load_state_dict(torch.load(
                            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

                    # Set Dataloader
                    vid_path, vid_writer = None, None
                    if webcam:
                        view_img = check_imshow()
                        cudnn.benchmark = True  # set True to speed up constant image size inference
                        dataset = LoadStreams(
                            source, img_size=imgsz, stride=stride)
                    else:
                        dataset = LoadImages(
                            source, img_size=imgsz, stride=stride)

                    # Get names and colors
                    names = model.module.names if hasattr(
                        model, 'module') else model.names
                    colors = [[random.randint(0, 255)
                               for _ in range(3)] for _ in names]

                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                            next(model.parameters())))  # run once
                    old_img_w = old_img_h = imgsz
                    old_img_b = 1

                    t0 = time.time()
                    ###################################
                    startTime = 0
                    ###################################
                    for path, img, im0s, vid_cap in dataset:
                        img = torch.from_numpy(img).to(device)
                        img = img.half() if half else img.float()  # uint8 to fp16/32
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)

                        # Warmup
                        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                            old_img_b = img.shape[0]
                            old_img_h = img.shape[2]
                            old_img_w = img.shape[3]
                            for i in range(3):
                                model(img, augment=opt.augment)[0]

                        # Inference
                        t1 = time_synchronized()
                        pred = model(img, augment=opt.augment)[0]
                        t2 = time_synchronized()

                        # Apply NMS
                        pred = non_max_suppression(
                            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                        t3 = time_synchronized()

                        # Apply Classifier
                        if classify:
                            pred = apply_classifier(pred, modelc, img, im0s)

                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            if webcam:  # batch_size >= 1
                                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                                ), dataset.count
                            else:
                                p, s, im0, frame = path, '', im0s, getattr(
                                    dataset, 'frame', 0)

                            p = Path(p)  # to Path
                            save_path = str(save_dir / p.name)  # img.jpg
                            txt_path = str(save_dir / 'labels' / p.stem) + \
                                ('' if dataset.mode ==
                                 'image' else f'_{frame}')  # img.txt
                            # normalization gain whwh
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(
                                    img.shape[2:], det[:, :4], im0.shape).round()

                                # Print results
                                for c in det[:, -1].unique():
                                    # detections per class
                                    n = (det[:, -1] == c).sum()
                                    # add to string
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                                dets_to_sort = np.empty((0, 6))
                                # NOTE: We send in detected object class too
                                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                                    dets_to_sort = np.vstack((dets_to_sort,
                                                              np.array([x1, y1, x2, y2, conf, detclass])))

                                if opt.track:

                                    tracked_dets = sort_tracker.update(
                                        dets_to_sort, opt.unique_track_color)
                                    tracks = sort_tracker.getTrackers()

                                    # draw boxes for visualization
                                    if len(tracked_dets) > 0:
                                        bbox_xyxy = tracked_dets[:, :4]
                                        identities = tracked_dets[:, 8]
                                        categories = tracked_dets[:, 4]
                                        confidences = None

                                        if opt.show_track:
                                            # loop over tracks
                                            for t, track in enumerate(tracks):

                                                track_color = colors[int(
                                                    track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                                # Extracting the centroids for the track
                                                centroids = [(int(coord[0]), int(coord[1]))
                                                             for coord in track.centroidarr]

                                                # Draw lines connecting centroids
                                                for i in range(len(centroids) - 1):
                                                    cv2.line(
                                                        im0, centroids[i], centroids[i + 1], track_color, thickness=opt.thickness)

                                            # track_color = colors[int(
                                            #     track.detclass)] if not opt.unique_track_color else sort_tracker.color_list[t]

                                            # [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            #                 int(track.centroidarr[i][1])),
                                            #           (int(track.centroidarr[i+1][0]),
                                            #            int(track.centroidarr[i+1][1])),
                                            #           track_color, thickness=opt.thickness)
                                            #  for i, _ in enumerate(track.centroidarr)
                                            #  if i < len(track.centroidarr)-1]
                                else:
                                    bbox_xyxy = dets_to_sort[:, :4]
                                    identities = None
                                    categories = dets_to_sort[:, 5]
                                    confidences = dets_to_sort[:, 4]

                                im0 = draw_boxes(im0, bbox_xyxy, identities,
                                                 categories, confidences, names, colors)
                            if dataset.mode != 'image' and opt.show_fps:
                                currentTime = time.time()

                                fps = 1/(currentTime - startTime)
                                startTime = currentTime
                                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70),
                                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                            # Save results (image with detections)
                            if save_img:
                                if dataset.mode == 'image':
                                    cv2.imwrite(save_path, im0)
                                    print(
                                        f" The image with the result is saved in: {save_path}")
                                else:  # 'video' or 'stream'
                                    if vid_path != save_path:  # new video
                                        vid_path = save_path
                                        if isinstance(vid_writer, cv2.VideoWriter):
                                            vid_writer.release()  # release previous video writer
                                        if vid_cap:  # video
                                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                            w = int(vid_cap.get(
                                                cv2.CAP_PROP_FRAME_WIDTH))
                                            h = int(vid_cap.get(
                                                cv2.CAP_PROP_FRAME_HEIGHT))
                                        else:  # stream
                                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                                            save_path += '.mp4'
                                        vid_writer = cv2.VideoWriter(
                                            save_path, cv2.VideoWriter_fourcc(*'H264'), fps, (w, h))
                                    vid_writer.write(im0)

                                    processed_image_container.image(
                                        im0, channels="BGR", use_column_width=True)

                    image_bytes = open(save_path, 'rb').read()

                    processed_image_container.image(image_bytes)
                    if save_txt or save_img:
                        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                        # print(f"Results saved to {save_dir}{s}")

                    # print(f'Done. ({time.time() - t0:.3f}s)')

                if __name__ == '__main__':
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--weights', nargs='+', type=str,
                                        default='yolov7.pt', help='model.pt path(s)')
                    # file/folder, 0 for webcam
                    parser.add_argument('--source', type=str,
                                        default='inference/images', help='source')
                    parser.add_argument('--img-size', type=int, default=640,
                                        help='inference size (pixels)')
                    parser.add_argument('--conf-thres', type=float,
                                        default=0.25, help='object confidence threshold')
                    parser.add_argument('--iou-thres', type=float,
                                        default=0.45, help='IOU threshold for NMS')
                    parser.add_argument('--device', default='',
                                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                    parser.add_argument('--view-img', action='store_true',
                                        help='display results')
                    parser.add_argument('--save-txt', action='store_true',
                                        help='save results to *.txt')
                    parser.add_argument('--save-conf', action='store_true',
                                        help='save confidences in --save-txt labels')
                    parser.add_argument('--nosave', action='store_true',
                                        help='do not save images/videos')
                    parser.add_argument('--classes', nargs='+', type=int,
                                        help='filter by class: --class 0, or --class 0 2 3')
                    parser.add_argument('--agnostic-nms', action='store_true',
                                        help='class-agnostic NMS')
                    parser.add_argument('--augment', action='store_true',
                                        help='augmented inference')
                    parser.add_argument('--update', action='store_true',
                                        help='update all models')
                    parser.add_argument('--project', default='runs/detect',
                                        help='save results to project/name')
                    parser.add_argument('--name', default='exp',
                                        help='save results to project/name')
                    parser.add_argument('--exist-ok', action='store_true',
                                        help='existing project/name ok, do not increment')
                    parser.add_argument('--no-trace', action='store_true',
                                        help='don`t trace model')

                    parser.add_argument('--track', action='store_true',
                                        help='run tracking')
                    parser.add_argument('--show-track', action='store_true',
                                        help='show tracked path')
                    parser.add_argument(
                        '--show-fps', action='store_true', help='show fps')
                    parser.add_argument('--thickness', type=int, default=2,
                                        help='bounding box and font size thickness')
                    parser.add_argument('--seed', type=int, default=1,
                                        help='random seed to control bbox colors')
                    parser.add_argument('--nobbox', action='store_true',
                                        help='don`t show bounding box')
                    parser.add_argument('--nolabel', action='store_true',
                                        help='don`t show label')
                    parser.add_argument('--unique-track-color', action='store_true',
                                        help='show each track in unique color')

                    with open("temp_image.jpeg", "wb") as f:
                        f.write(image.read())

                    opt = argparse.Namespace(
                        weights='best_2.pt',
                        source="temp_image.jpeg",
                        no_trace=True,
                        view_img=True,
                        nosave=False,
                        track=True,
                        classes=selected_values,
                        show_track=True,
                        img_size=448,
                        seed=1,
                        update=False,
                        save_txt=False,
                        project='runs/detect',  # Add the project attribute here
                        name='exp',
                        exist_ok=True,
                        device="cpu",
                        augment=False,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        agnostic_nms=False,
                        show_fps=False,
                        unique_track_color=False,
                        thickness=thickness_image,
                        nobbox=False,
                        nolabel=False
                    )

                    # print(opt)

                # Set random seed
                    np.random.seed(opt.seed)

                    # Initialize Sort tracker
                    sort_tracker = Sort(max_age=5,
                                        min_hits=2,
                                        iou_threshold=0.2)

                    # check_requirements(exclude=('pycocotools', 'thop'))

                    with torch.no_grad():
                        detect()

                # Open video file and read its bytes

                    # Display the MIME type
            else:
                st.info(
                    "Please upload an image and choose and choose thickness for bounding boxes.", icon="ℹ️")

        image_file_path = 'runs/detect/exp/temp_image.jpeg'
        image_file = open(image_file_path, 'rb')
        image_bytes = image_file.read()
        with st.container():  # Adjust width as needed
            # Display the video
            st.image(image_bytes)

        with open(image_file_path, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name="temp_image.jpeg",
                mime="image/jpeg"
            )
