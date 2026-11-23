from constants import FRAME_POINTS, MAP_POINTS
import cv2 as cv

def determine_if_q1(box):
    return True if box[0] < FRAME_POINTS[2][0] and box[0] > FRAME_POINTS[0][0] and box[1] < FRAME_POINTS[-1][1] else False

def determine_if_q2(box):
    return True if box[0] < FRAME_POINTS[2][0] and box[0] > FRAME_POINTS[0][0] and box[1] > FRAME_POINTS[-1][1] else False

def determine_if_q3(box):
    return True if box[0] > FRAME_POINTS[6][0] and box[0] < FRAME_POINTS[0][0] and box[1] > FRAME_POINTS[-1][1] else False

def determine_if_q4(box):
    return True if box[0] > FRAME_POINTS[6][0] and box[0] < FRAME_POINTS[0][0] and box[1] < FRAME_POINTS[-1][1] else False

def calculate_ratios(map_pt1, map_pt2, frame_pt1, frame_pt2):
    x = abs(map_pt2[0] - map_pt1[0]) / abs(frame_pt2[0] - frame_pt1[0])
    y = abs(map_pt2[1] - map_pt1[1]) / abs(frame_pt2[1] - frame_pt1[1])
    return [x, y]

def distance_away_map_coords(box, ratio_x, ratio_y, reference_point):
    reference_x = reference_point[0]
    box_x = box[0]
    refernce_y = reference_point[1]
    box_y = box[1]
    distance_x = abs(box_x - reference_x)
    distance_y = abs(box_y - refernce_y)
    distance_in_map_x = distance_x * ratio_x
    distance_in_map_y = distance_y * ratio_y

    return [distance_in_map_x, distance_in_map_y]


def get_data_for_quadrant(box):

    map_reference_pt1 = []
    map_reference_pt2 = []
    frame_reference_pt1 = []
    frame_reference_pt2 = []
    reference_point = []
    assigned  : bool = False
    quadrant : str = ""
                
    if determine_if_q1(box):
        map_reference_pt1 = MAP_POINTS[0]
        map_reference_pt2 = MAP_POINTS[2]
        frame_reference_pt1 = FRAME_POINTS[0]
        frame_reference_pt2 = FRAME_POINTS[2]
        reference_point = FRAME_POINTS[0]
        assigned = True
        quadrant = "q1"
    elif determine_if_q2(box):
       map_reference_pt1 = MAP_POINTS[2]
       map_reference_pt2 = MAP_POINTS[4]
       frame_reference_pt1 = FRAME_POINTS[2]
       frame_reference_pt2 = FRAME_POINTS[4]
       reference_point = FRAME_POINTS[2]
       assigned = True
       quadrant = "q2"
    elif determine_if_q3(box):
        map_reference_pt1 = MAP_POINTS[4]
        map_reference_pt2 = MAP_POINTS[6]
        frame_reference_pt1 = FRAME_POINTS[4]
        frame_reference_pt2 = FRAME_POINTS[6]
        reference_point = FRAME_POINTS[4]
        assigned = True
        quadrant = "q3"
    else:
       map_reference_pt1 = MAP_POINTS[6]
       map_reference_pt2 = MAP_POINTS[8]
       frame_reference_pt1 = FRAME_POINTS[6]
       frame_reference_pt2 = FRAME_POINTS[8]
       reference_point = FRAME_POINTS[6]
       assigned = True
       quadrant = "q4"
    
    return map_reference_pt1, map_reference_pt2, frame_reference_pt1, frame_reference_pt2, reference_point, assigned, quadrant
