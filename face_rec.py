import os
import cv2
from cv2 import Mat
import dlib
import numpy as np
import face_recognition
import face_recognition as fr


# def wearing_glasses(img: Mat):
#     detector = dlib.get_frontal_face_detector()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 1)

#     for i, face_rect in enumerate(rects):
#         left = face_rect.left() 
#         top = face_rect.top()
#         width = face_rect.right() - left
#         height = face_rect.bottom() - top
#         # print(img[top + 10:top+height-100, left + 30: left+width - 20])
#         '''Cropping an another frame from the detected face rectangle'''
#         frame_crop = img[top + 10:top+height-100, left + 30: left+width - 20]

#         # Smoothing the cropped frame
#         img_blur = cv2.GaussianBlur(np.array(img),(5,5), sigmaX=1.7, sigmaY=1.7)
#         # Filterting the cropped frame through the canny filter
#         edges = cv2.Canny(image = img_blur, threshold1=100, threshold2=200)

#         # Center Strip
#         edges_center: np.ndarray = edges.T[(int(len(edges.T)/2))]
#         # 255 represents white edges. If any white edges are detected
#         # in the desired place, it will show 'Glass is Present' message
#         cv2.imshow('Edge', edges_center)
#         unique, counts = np.unique(edges_center, return_counts=True)
#         if dict(zip(unique, counts))[255] > 1:
#             return True
#         else:
#             return False

def get_known_faces() -> dict:
    print('getting faces')
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for root, directory, file_names in os.walk("./faces"):
        for file_name in file_names:
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                face = fr.load_image_file("faces/" + file_name)
                face_loc = fr.face_locations(face)
                print('encoding known faces')
                encoding = fr.face_encodings(face, face_loc)[0]
                encoded[file_name.split(".")[0]] = encoding
    print(encoded)
    return encoded


# def unknown_image_encoded(img):

#     """
#     encode a face given the file name
#     """
#     face = fr.load_image_file("faces/" + img)
#     encoding = fr.face_encodings(face)[0]

#     return encoding


def classify_face(im: str, faces: dict):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    print('getting test face')
    img = cv2.imread(im, 1)
    print('getting unknown face location')
    face_locations = face_recognition.face_locations(img)
    print('encoding unknown face')
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
    # is_wearing_glasses = wearing_glasses(img)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        # if is_wearing_glasses:
        print('matching faces')
        matches = face_recognition.compare_faces(faces_encoded, face_encoding, tolerance=0.588)
        # else:
        # matches = face_recognition.compare_faces(faces_encoded, face_encoding, tolerance=0.5)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            print('drawing faces')
            # Draw a box around the face
            cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names

        # print(classify_face("test.jpg"))


if __name__ == '__main__':
    faces = get_known_faces()
    classify_face("test.jpg", faces)
