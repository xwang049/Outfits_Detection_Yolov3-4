#!/usr/bin/env python3

import argparse
import numpy as np
import sys
import cv2
# from PIL import Image

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

import time

# Classifier input size.
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    # image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.ANTIALIAS)
    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(np.array(image, dtype=np.float32, order='C'), (2, 0, 1))
    # VGG.
    image[0] -= 103.939
    image[1] -= 116.779
    image[2] -= 123.68
    # Inception.
    # image /= 127.5
    # image -= 1
    
    # Yolov3/V4.
    # image /= 255.0

    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='color',
                        help='Inference model name, default yolov4.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001.')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        default='',
                        help='Write output into file instead of displaying it.')
    parser.add_argument('-c',
                        '--confidence',
                        type=float,
                        required=False,
                        default=0.3,
                        help='Confidence threshold for detected objects, default 0.8.')
    parser.add_argument('-n',
                        '--nms',
                        type=float,
                        required=False,
                        default=0.4,
                        help='Non-maximum suppression threshold for filtering raw boxes, default 0.5.')
    parser.add_argument('-f',
                        '--fps',
                        type=float,
                        required=False,
                        default=24.0,
                        help='Video output fps, default 24.0 FPS.')
    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics.')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output.')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server.')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none.')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none.')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none.')

    FLAGS = parser.parse_args()

    # Create server context.
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("Context creation failed: " + str(e))
        sys.exit()

    # Health check.
    if not triton_client.is_server_live():
        print("FAILED: is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED: is_server_ready")
        sys.exit(1)
    
    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED: is_model_ready")
        sys.exit(1)
    # To display the meta.
    # python3 client.py image fashion.jpg -i
    if FLAGS.model_info:
        # Model metadata.
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration.
        try:
            config = triton_client.get_model_config(FLAGS.model)
            if not (config.config.name == FLAGS.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
    # IMAGE MODE.
    
    print("Running in 'image' mode")
    if not FLAGS.input:
        print("FAILED: no input image")
        sys.exit(1)
    
    inputs = []
    outputs = []
    # Specify the input size for classifier.
    inputs.append(grpcclient.InferInput('input_1', [1, 3, INPUT_HEIGHT, INPUT_WIDTH], "FP32"))
    # Specify the output nodes.
    outputs.append(grpcclient.InferRequestedOutput('color0'))
 

    print("Creating buffer from image file...")
    
    input_image = cv2.imread(str(FLAGS.input))
    # input_image = Image.open(str(FLAGS.input)) # .convert('RGB')

    if input_image is None:
        print(f"FAILED: could not load input image {str(FLAGS.input)}")
        sys.exit(1)
    input_image_buffer = preprocess(input_image)

    # For the shape.
    # input_image = np.array(input_image.convert('RGB'))

    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    print("Invoking inference...")
    results = triton_client.infer(model_name=FLAGS.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=FLAGS.client_timeout)
    if FLAGS.model_info:
        statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
        if len(statistics.model_stats) != 1:
            print("FAILED: get_inference_statistics")
            sys.exit(1)
        print(statistics)
    
    outputs_res = [results.as_numpy('color0')] # neck_design.


    labels_path = ["labels/color.txt"]
    all_labels = []

    for label_file in labels_path:
        f = open(label_file, 'r')
        l = f.readlines()
        f.close()
        labels = []
        for t in l:
            labels.append(t.split()[0])
        all_labels.append(labels)
    
    classifier_output = len(outputs_res)

    for i in range(classifier_output):
        output = outputs_res[i]
        idx = np.argmax(output)
        print(all_labels[i], idx, output[0][idx], all_labels[i][idx])
    """
['Invisible', 'TurtleNeck', 'RuffleSemi-HighCollar', 'LowTurtleNeck', 'DrapedCollar'] 4 [[0.8871879]] DrapedCollar
['Invisible', 'HighWaist', 'Regular', 'Long', 'Micro', 'Knee', 'Midi', 'AnkleFloor'] 7 [[0.99978024]] AnkleFloor
['Invisible', 'Sleeveless', 'CupSleeves', 'ShortSleeves', 'ElbowSleeves', '3/4Sleeves', 'WristLength', 'LongSleeves', 'ExtraLongSleeves'] 8 [[0.6724836]] ExtraLongSleeves
['Invisible', 'Strapless', 'DeepVNeckline', 'Straight', 'VNeckline', 'SquareNeckline', 'OffShoulder', 'RoundNeckline', 'SweatHeartNeck', 'OneShoulderNeckline'] 0 [[0.29348585]] Invisible
['Invisible', 'Shirt', 'Peter', 'Puritan', 'Rib'] 0 [[0.91322947]] Invisible
['Invisible', 'ShortPant', 'MidLength', '3/4Length', 'CroppedPant', 'FullLength'] 0 [[0.9781059]] Invisible
['Invisible', 'Short', 'Knee', 'Midi', 'Ankle', 'Floor'] 5 [[0.99832827]] Floor
['Invisible', 'Notched', 'Collarless', 'ShawlCollar', 'PlusSizeShawl'] 0 [[0.99026287]] Invisible
    """
    

    # detected_objects = postprocess(result, input_image.shape[1], input_image.shape[0], FLAGS.confidence, FLAGS.nms)
    # print(f"Raw boxes: {int(result[0, 0, 0, 0])}")
    # print(f"Detected objects: {len(detected_objects)}")

    # for box in detected_objects:
    #     # print(f"{COCOLabels(box.classID).name}: {box.confidence}")
    #     input_image = render_box(input_image, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
    #     size = get_text_size(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", normalised_scaling=0.6)
    #     input_image = render_filled_box(input_image, (box.x1-3, box.y1-3, box.x1+size[0], box.y1+size[1]), color=(220, 220, 220))
    #     input_image = render_text(input_image, f"{COCOLabels(box.classID).name}: {box.confidence:.2f}", (box.x1, box.y1), color=(30, 30, 30), normalised_scaling=0.5)

    # if FLAGS.out:
    #     cv2.imwrite(FLAGS.out, input_image)
    #     print(f"Saved result to {FLAGS.out}")
    # else:
    #     cv2.imshow('image', input_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

