import cv2
import numpy as np
from collections import deque

class backgroundExt:
    def __init__(self,width, height, maxlenth):
        self.maxlenth = maxlenth
        self.width = width
        self.height = height
        self.buffer = deque(maxlen=maxlenth)
        self.new_frame = None
        self.bg = None

    def cal_background(self):
        self.bg = np.zeros((self.height, self.width), dtype="uint16")
        for i in self.buffer:
            self.bg += i
        self.bg //= len(self.buffer)

    def update_background(self, old_frame, new_frame):
        self.bg -= old_frame//self.maxlenth
        self.bg += new_frame//self.maxlenth

    def update_Frame(self, frame):
        if len(self.buffer) < self.maxlenth:
            self.buffer.append(frame)
            self.cal_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_bg(self):
        return self.bg.astype("uint8")

    def get_bg_mask(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        bg_buffer.update_Frame(gray)
        moving_abs_diff = cv2.absdiff(self.get_bg(), gray)
        _, moving_abs_mask = cv2.threshold(moving_abs_diff, 10, 255, cv2.THRESH_BINARY)
        dilated_mask = cv2.dilate(moving_abs_mask, None, iterations=3)
        return dilated_mask

    def get_fg_mask(self,frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        moving_abs_diff = cv2.absdiff(self.get_bg(), gray)
        _, moving_abs_mask = cv2.threshold(moving_abs_diff, 50, 255, cv2.THRESH_BINARY)
        dilated_mask = cv2.dilate(moving_abs_mask, None, iterations=3)
        return dilated_mask

class obj:
    def __init__(self,width, height, size=50):
        self.width = width
        self.height = height
        self.size = size
        self.img = cv2.imread("flappy.png")
        self.img = cv2.resize(self.img, (self.size, self.size))
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        self.x = np.random.randint(0, self.width - self.size)
        self.y = 0
        self.speed = 5
        self.score = 0

    def insert(self,frame):
        roi = frame[self.y: self.size + self.y, self.x: self.size + self.x]
        roi[np.where(self.mask)] = 0
        roi += self.img

    def change_pos(self, fg_mask, frame):
        self.y = self.y + self.speed
        if self.y + self.size >= self.height:
            self.y = 0
            self.x = np.random.randint(0, self.width - self.size)
            self.score += 1
        roi = fg_mask[self.y: self.size + self.y, self.x: self.size + self.x]
        check = np.any(np.where(roi))
        if check == True:
            self.score -= 1
            self.y = 0
            self.x = np.random.randint(0, self.width - self.size)
            frame[:,:,2] = 255

    def display_score(self, frame):
        cv2.putText(frame, "SCORE:" + str(self.score), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

h=480
w=640
scale = 2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

bg_buffer = backgroundExt(w,h,maxlenth=10)
game = obj(w,h)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    fg_mask = bg_buffer.get_bg_mask(frame)
    cv2.putText(frame, "HIDE FROM THE FRAME AND PRESS S", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('s'):
        break


while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    fg_mask = bg_buffer.get_fg_mask(frame)
    game.insert(frame)
    game.change_pos(fg_mask, frame)
    game.display_score(frame)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    if cv2.waitKey(1) == ord('q'):
        break