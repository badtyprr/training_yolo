# Bing Image Search API v7 - Image Scraper Application
# Adapted from Adrian Rosebrock's Bing image scraper
# Source: https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/

# Python libraries
import os
import logging
import argparse
import time
# 3rd Party Libraries
import cv2
import numpy as np
import requests
# User libraries
from keys.azure import subscription_key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
    help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=False,
    help="path to output directory of images")
ap.add_argument("-m", "--max", required=False,
    help="maximum number of search results")
ap.add_argument("-g", "--group", required=False,
    help="group size to process query")
ap.add_argument("-t", "--timeout", required=False,
    help="timeout in seconds to wait for an image to download")
ap.add_argument("--width", required=False,
    help="minimum required width")
ap.add_argument("--height", required=False,
    help="minimum required height")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

"""
# Load the image classifier
"""
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(int(time.time()))
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

"""
# Azure parameters
"""
# NOTE: you need a subscription key from Azure, place it in keys/azure.py with the identifier subscription_key
# Source: https://portal.azure.com
assert subscription_key
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# Global Variables
if args["max"]:
    MAX_RESULTS = int(args["max"])
else:
    MAX_RESULTS = 250
if args["group"]:
    GROUP_SIZE = int(args["group"])
else:
    GROUP_SIZE = 50
if args["timeout"]:
    TIMEOUT = int(args["timeout"])
else:
    TIMEOUT = 30
if args["width"]:
    MIN_WIDTH = int(args["width"])
else:
    MIN_WIDTH = 0
if args["height"]:
    MIN_HEIGHT = int(args["height"])
else:
    MIN_HEIGHT = 0
if args["output"]:
    OUTPUT_DIR = args["output"]
else:
    OUTPUT_DIR = os.path.join('output', args["query"].replace(' ', '_').replace('\"', '').replace(':', '-').replace('.', 'p'))[:64]

"""
# Possible exceptions
"""
exception_set = [
    IOError, FileNotFoundError,
    requests.RequestException, requests.HTTPError,
    requests.ConnectionError, requests.Timeout,
    requests.exceptions.SSLError, requests.exceptions.ConnectTimeout
]
"""
# Possible image extensions
"""
extensions = [
    'jpg', 'jpeg', 'jpe', 'png', 'bmp',
    'pbm', 'pgm', 'ppm', 'sr', 'ras',
    'jp2', 'tiff', 'tif'
]
"""
# Create output directory
"""
# Source: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

"""
# Search query
"""
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
params = {"q": args["query"], "imageType": "photo"}
logging.info("searching Bing API for '{}'".format(term))

try:
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    estNumResults = min(search_results["totalEstimatedMatches"], MAX_RESULTS)
    logging.info("{} total results for '{}'".format(estNumResults, term))

    # Download images
    total = 0
    # Loop over total results in GROUP_SIZE batches
    for offset in range(0, estNumResults, GROUP_SIZE):
        logging.info("making request for group {}-{} of {}...".format(
            offset,
            offset+GROUP_SIZE,
            estNumResults
        ))
        params["offset"] = offset
        search = requests.get(search_url, headers=headers, params=params)
        search.raise_for_status()
        results = search.json()
        logging.info("saving images for group {}-{} of {}...".format(
            offset,
            offset+GROUP_SIZE,
            estNumResults
        ))
        # Loop over group batch
        for v in results["value"]:
            try:
                logging.info("fetching: {}".format(v["contentUrl"]))
                try:
                    r = requests.get(v["contentUrl"], timeout=TIMEOUT)
                except requests.ReadTimeout:
                    continue
                ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                ext = ext.lower()

                if v["width"] < MIN_WIDTH or v["height"] < MIN_HEIGHT:
                    logging.warning("{}x{} is smaller than {}x{}, skipping: {}".format(
                        v["width"], v["height"],
                        MIN_WIDTH, MIN_HEIGHT,
                        v["contentUrl"])
                    )
                    continue
                # Check for valid extensions
                elif ext[1:] not in extensions:

                    ext = ext[:4]
                    if ext[1:] not in extensions:
                        # default to .jpg
                        ext = '.jpg'
                        logging.warning("invalid extension, assigning extension as jpg: {}".format(v["contentUrl"]))
                    else:
                        logging.warning("invalid extension, truncating extension: {}".format(v["contentUrl"]))
                # path in format 'outputPath/number.ext'
                p = os.path.sep.join(
                    [OUTPUT_DIR,
                     "{}{}".format(str(total).zfill(8), ext)]
                )
                # Write to disk
                f = open(p, "wb")
                f.write(r.content)
                f.close()
            except Exception as e:
                if type(e) in exception_set:
                    logging.warning("encountered exception: {}, skipping: {}".format(e, v["contentUrl"]))
                    continue
                else:
                    raise e
            """
            # verify it's a good image
            """
            image = cv2.imread(p)
            if image is None:
                logging.warning("deleting corrupted image: {}".format(p))
                os.remove(p)
                continue
            (H, W) = image.shape[:2]
            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(
                image,
                1 / 255.0,  # sigma
                (416, 416),  # Resize to X by Y
                swapRB=True,  # BGR to RGB
                crop=False  # cropping
            )
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                    args["threshold"])

            # Test to see if at least one person was detected
            if not any([LABELS[c] == 'person' for c in classIDs]):
                logging.warning("deleting image because no people found: {}".format(p))
                os.remove(p)
                continue

            # Found a good image, go to the next
            total += 1
except requests.HTTPError as e:
    # Sometimes you'll get this error if you entered the wrong subscription key
    # Source: https://blogs.msdn.microsoft.com/kwill/2017/05/17/http-401-access-denied-when-calling-azure-cognitive-services-apis/
    print(e)
