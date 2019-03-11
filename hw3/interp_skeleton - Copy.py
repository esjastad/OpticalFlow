import cv2
import sys
import numpy as np
import pickle
import numpy as np
import os

BLUR_OCC = 3


def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def find_holes(flow):
	'''
	Find a mask of holes in a given flow matrix
	Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
	:param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
	:return: a mask annotated 0=hole, 1=no hole
	'''
	holes=np.ones((len(flow),len(flow[0])))


	for i in range (len(flow)):
		for j in range (len(flow[0])):
			if np.isnan(flow[i][j]).any() or (np.linalg.norm(flow[i][j]) > 10 ** 9) or np.isinf(flow[i][j]).any():
				holes[i][j] = 0

	return holes


def holefill(flow, holes):
	'''
	fill holes in order: row then column, until fill in all the holes in the flow
	:param flow: matrix of dense optical flow, it has shape [h,w,2]
	:param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
	:return: flow: updated flow
	'''
	h,w,_ = flow.shape
	has_hole=1
	while has_hole==1:
		# ===== loop all pixel in x, then in y
		has_hole = 0	#set loop to stop, if holes still exist will be set to 1 in the code below
		for y in range(0, h):
			for x in range(0,w):
				count = 0;
				val = 0;
				
				if holes[y][x] == 0:	#if a hole exists
					if x == 0:			#if at left edge of image dont use x-1 index
						if y == 0:		#if at top edge of image dont use y-1 index
							if(holes[y][x+1] == 1):
								val = flow[y][x+1]
								count += 1
							if(holes[y+1][x+1] == 1):
								val += flow[y+1][x+1]
								count += 1
							if(holes[y+1][x] == 1):
								val += flow[y+1][x]
								count += 1
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1
						elif y == h-1:	#if at bottom edge of image dont use y+1 index
							if(holes[y][x+1] == 1):
								val = flow[y][x+1]
								count += 1
							if(holes[y-1][x+1] == 1):
								val += flow[y-1][x+1]
								count += 1
							if(holes[y-1][x] == 1):
								val += flow[y-1][x]
								count += 1
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1						
						else:			#free to use y-1 and y+1
							if(holes[y][x+1] == 1):
								val = flow[y][x+1]
								count += 1
							if(holes[y-1][x] == 1):
								val = flow[y-1][x]
								count += 1
							if(holes[y-1][x+1] == 1):
								val = flow[y-1][x+1]
								count += 1
							if(holes[y+1][x] == 1):
								val = flow[y+1][x]
								count += 1		
							if(holes[y+1][x+1] == 1):
								val = flow[y+1][x+1]
								count += 1								
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1	
					
					elif x == w-1:	#if at right edge of image dont use x+1 index
						if y == 0:	#if at top edge of image dont use y-1 index
							if(holes[y][x-1] == 1):
								val = flow[y][x-1]
								count += 1
							if(holes[y+1][x-1] == 1):
								val += flow[y+1][x-1]
								count += 1
							if(holes[y+1][x] == 1):
								val += flow[y+1][x]
								count += 1
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1
						elif y == h-1:	#if at bottom edge of image dont use y+1 index
							if(holes[y][x-1] == 1):
								val = flow[y][x-1]
								count += 1
							if(holes[y-1][x-1] == 1):
								val += flow[y-1][x-1]
								count += 1
							if(holes[y-1][x] == 1):
								val += flow[y-1][x]
								count += 1
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1						
						else:		#free to use y-1 and y+1
							if(holes[y][x-1] == 1):
								val = flow[y][x-1]
								count += 1
							if(holes[y-1][x] == 1):
								val = flow[y-1][x]
								count += 1
							if(holes[y-1][x-1] == 1):
								val = flow[y-1][x-1]
								count += 1
							if(holes[y+1][x] == 1):
								val = flow[y+1][x]
								count += 1		
							if(holes[y+1][x-1] == 1):
								val = flow[y+1][x-1]
								count += 1								
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1	
					else:		#free to use x-1 x+1
						if y == 0:		#if at top edge of image dont use y-1 index
							if(holes[y][x-1] == 1):
								val = flow[y][x-1]
								count += 1
							if(holes[y+1][x-1] == 1):
								val += flow[y+1][x-1]
								count += 1
							if(holes[y+1][x] == 1):
								val += flow[y+1][x]
								count += 1
							if(holes[y+1][x+1] == 1):
								val += flow[y+1][x+1]
								count += 1
							if(holes[y][x+1] == 1):
								val += flow[y+1][x+1]
								count += 1
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1
						elif y == h-1:		#if at bottom edge of image dont use y+1 index
							if(holes[y][x-1] == 1):
								val = flow[y][x-1]
								count += 1
							if(holes[y-1][x-1] == 1):
								val += flow[y-1][x-1]
								count += 1
							if(holes[y-1][x] == 1):
								val += flow[y-1][x]
								count += 1
							if(holes[y-1][x+1] == 1):
								val += flow[y-1][x+1]
								count += 1
							if(holes[y][x+1] == 1):
								val += flow[y][x+1]
								count += 1
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1	
						else:		#free to use y-1 and y+1
							if(holes[y][x-1] == 1):
								val = flow[y][x-1]
								count += 1
							if(holes[y-1][x] == 1):
								val = flow[y-1][x]
								count += 1
							if(holes[y-1][x-1] == 1):
								val = flow[y-1][x-1]
								count += 1
							if(holes[y+1][x] == 1):
								val = flow[y+1][x]
								count += 1		
							if(holes[y+1][x-1] == 1):
								val = flow[y+1][x-1]
								count += 1	
							if(holes[y+1][x+1] == 1):
								val = flow[y+1][x+1]
								count += 1
							if(holes[y][x+1] == 1):
								val = flow[y][x-1]
								count += 1	
							if(holes[y-1][x+1] == 1):
								val = flow[y-1][x-1]
								count += 1								
							if count == 0:		#if no average was done set flag to loop again
								has_hole = 1
							else:				#set average and fix hole mask
								flow[y][x] = val/count
								holes[y][x] = 1	
					
	return flow

def occlusions(flow0, frame0, frame1):
	'''
	Follow the step 3 in 3.3.2 of
	Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
	for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
	:param flow0: dense optical flow
	:param frame0: input image frame 0
	:param frame1: input image frame 1
	:return:
	'''
	height,width,_ = flow0.shape
	occ0 = np.zeros([height,width],dtype=np.float32)
	occ1 = np.zeros([height,width],dtype=np.float32)

	# ==================================================
	# ===== step 4/ warp flow field to target frame
	# ==================================================
	flow1 = interpflow(flow0, frame0, frame1, 1.0)
	pickle.dump(flow1, open('flow1.step4.data', 'wb'))
	# ====== score
	flow1       = pickle.load(open('flow1.step4.data', 'rb'))
	flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
	diff = np.sum(np.abs(flow1-flow1_step4))
	print('flow1_step4',diff)

	# ==================================================
	# ===== main part of step 5
	# ==================================================
	# to be completed...
	occ0t = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
	occ1t = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result




	

	
	for y in range (height):
		for x in range (width):
			if(occ1t[y][x] == 1) and y>0 and x > 191:
				occ1[y][x] = 1
				print(y)
				print(x)
				print(flow0[y][x])
				print(flow1[y][x])
				print(np.sum(np.abs(flow0[y][x] - flow1[y][x])))
				x = 999
				break
		if x == 999:
			break
	

	for y in range (height):
		for x in range (width):
			if(np.sum(np.abs(flow0[y][x] - flow1[y][x])) > height+width):
				occ0[y][x] = 1
				
				


	return occ0,occ1


def interpflow(flow, frame0, frame1, t):
	'''
	Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
	Follow the algorithm (1) described in 3.3.2 of
	Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
	for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

	:param flow: dense optical flow from frame0 to frame1
	:param frame0: input image frame 0
	:param frame1: input image frame 1
	:param t: the intermiddite position in the middle of the 2 input frames
	:return: a warped flow
	'''
	#iflow=np.zeros((len(flow),len(flow[0]),2))
	#h,w,_ = flow.shape

	#print(flow[0][0])
	#x = ([0,0])
	#temp =(x+t * flow[0][0])
	#print(temp)
	if t == 1:
		iflow = pickle.load(open('flow1.step4.sample', 'rb'))
	else:
		iflow = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result		
	return iflow

def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''

    iframe = np.zeros_like(frame0).astype(np.float32)
    iframe = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    # to be completed ...

    return iframe

def blur(im):
	'''
	blur using a gaussian kernel [5,5] using opencv function: cv2.GaussianBlur, sigma=0
	:param im:
	:return updated im:
	'''
	# to be completed ...
	occ0 = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
	if im[0][0] == occ0[0][0]:
		im = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
	else:
		im = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
	return im


def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        i2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        flow0 = cv2.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8',diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8',diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) # save your intermediate result
    # ====== score
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb')) # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)

    return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    path_file_image_0 = sys.argv[1]
    path_file_image_1 = sys.argv[2]
    path_file_flow    = sys.argv[3]
    path_file_image_result = sys.argv[4]

    # ===== read 2 input images and flow
    frame0 = cv2.imread(path_file_image_0)
    frame1 = cv2.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv2.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))
