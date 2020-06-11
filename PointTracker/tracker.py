import numpy as np
import cv2 as cv
import argparse

#Get position of the mouse
def getPos(event,y,x,flags,param):
    global count,position
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(x,y)
        position.append([[y,x]])
        count+=1

#Init
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
parser.add_argument('tracklines', type=int, help='track the points?')
parser.add_argument('npoints', type=int, help='number of points')

#Get the argument from the console line
args = parser.parse_args()
draw_lines = args.tracklines
n_points = args.npoints

#Start the capture of video
cap = cv.VideoCapture(args.image)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (30,30),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame
ret, old_frame = cap.read()
old_frame = cv.rotate(old_frame, cv.ROTATE_90_CLOCKWISE) #I rotate it 90º to se it vertical, you may have to change this
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

win_scale = 2 #Scale for the visualization size/scale

#Create window and set the callback function for the double left-click
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (int(frame_height/win_scale),int(frame_width/win_scale)))
cv.setMouseCallback('frame',getPos)


##START OF THE PROGRAM
print("PICK A POINT IN THE IMAGE")
#Init of points
p0 = None
ix,iy = None,None
position = []
count = 0
#-
#We ask for the n points in the image to track
while(count < n_points):
    cv.resizeWindow('frame', (int(frame_height/2.5),int(frame_width/2.5)))
    cv.imshow('frame',old_frame)
    k = cv.waitKey(10) #Add some delay in the loop
    if k == 27: #If esc is pressed we exit the process
        break
    for point in position:
        old_frame = cv.circle(old_frame,(point[0][0],point[0][1]),10,color[count].tolist(),-1)

p0 = np.float32(position) #Then convert the points into a float32 array. That is the same structure as the UMat from OpenCV


# Create a mask to draw the points and trajectories
mask = np.zeros_like(old_frame)
#Create a Video write to save the output video
out = cv.VideoWriter('outputTr.avi',cv.VideoWriter_fourcc(*'XVID'),25.0,(frame_height,frame_width))
counter = 0
##PARAMETER
frames_to_process = 250
max_err = 10
##--
while(counter<frames_to_process): #We process x frames of the video
    active_points = p0.size/2
    counter+=1
    ret,frame = cap.read() #Read the next frame
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #create a gray copy for the optical flow
    
    #If we have any active points we calculate the optical flow.
    if(active_points > 0):
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        #Select good points with an erros smaller than max_err
        good_new = p1[st==1 & (err < max_err)]
        good_old = p0[st==1 & (err < max_err)]
        print(st,err)

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            a,b,c,d = int(a),int(b),int(c),int(d)
            if draw_lines:
                mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2) #trajectories
            frame = cv.circle(frame,(a,b),10,color[i].tolist(),-1) #current location
    
    # Create the current frame with the mask to display
    img = cv.add(frame,mask)
    out.write(img)
    cv.imshow('frame',img)
    
    k = cv.waitKey(30) & 0xff
    if k == 27: #If esc is pressed we exit the process
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    

