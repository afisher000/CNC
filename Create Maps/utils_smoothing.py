import cv2 as cv
import numpy as np
import utils_contours as uc
from numpy.linalg import norm

def convert_map_to_SVG(pngfile, svgfile, minArea=100, maxArea=1e5, troubleshooting=False):
    gray = cv.imread(pngfile, cv.IMREAD_GRAYSCALE)
    white = np.full_like(gray, 255, np.uint8)

    # Find contours, remove small/large areas
    all_contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv.contourArea(c) for c in all_contours])
    contours = np.delete(
        np.array(all_contours, dtype=object), 
        np.where((areas<minArea)|(areas>maxArea))[0],
        0
    )
    print(f'Reduced number of contours from {len(all_contours)} to {len(contours)}')

    # Smooth each contour
    smooth_contours = []
    for j, contour in enumerate(contours):
        # Get points along line, remove duplicates, and sort
        points = get_points_on_line(white, contour)
        points = remove_duplicate_points(points, min_dist=10)
        points = sort_line_points(points, max_sep=60, angle_penalty_thresh=0.1)

        # Downsample curve and save as new contour
        points = smooth_curve(points, 2)
        smooth_contour = build_contour_from_points(points, linewidth=2)
        smooth_contours.append(smooth_contour)

        if troubleshooting:
            print(j)
            white = np.full_like(gray, 255, np.uint8)  
            cv.drawContours(white, [contour], -1, 0, -1)
            uc.showImage(white)
            white = np.full_like(gray, 255, np.uint8)  
            white = cv.drawContours(white, [smooth_contour], -1, 0, -1)
            uc.showImage(white)


    print(f'Reduced total points from {sum(map(len, contours))} to {sum(map(len, smooth_contours))}')
    uc.save2SVG(svgfile, smooth_contours, white.shape)
    return

def sort_line_points(points, max_sep=20, theta_buffer = 0.05, max_theta=120, dist_power=2):
    # Start with first point, follow line in both directions
    # Append or insert into js array based on direction
    INF = 1e6
    first_direction = True

    # Start with 0th point, find nearest neighbor
    js = [0]
    dists = np.sum(np.square(points-points[0,:]), axis=1)
    dists[js] = INF
    next_j = np.argmin(dists)

    # Append to js and begin iterations
    js = [0, next_j]
    prev_j, cur_j = 0, next_j

    while True:
        linedir = points[cur_j,:] - points[prev_j,:]
        pointdirs = points - points[cur_j,:]
        dists = norm(pointdirs, axis=1)
        dists[js] = INF
        costheta = np.dot(pointdirs, linedir)/norm(linedir)/dists

        weighted_dists = (dists/max_sep)**dist_power * (1-costheta + theta_buffer)
        
        next_j = np.argmin(weighted_dists)
        is_too_far = dists[next_j]>max_sep
        is_backwards = costheta[next_j]<np.cos(max_theta*np.pi/180)
        if (not is_too_far) and (not is_backwards):
            if first_direction:
                js.append(next_j)  
            else:
                js.insert(0, next_j)
            prev_j, cur_j = cur_j, next_j
        # If invalid, switch direction or break
        else:
            if is_too_far:
                print('Too far')
            else:
                print('Is backwards')
            if first_direction:
                first_direction = False
                prev_j, cur_j = js[1], js[0]
            else:
                break
    return points[js,:]



def update_point(contour, point, dx, dy):
    f0 = cv.pointPolygonTest(contour, point, True)
    fx = cv.pointPolygonTest(contour, point + dx, True)
    fy = cv.pointPolygonTest(contour, point + dy, True)

    next_point = (point + np.array([fy-f0, fx-f0])).round().astype(np.int16)
    return next_point


def optimize_point(contour, point):
    dx = np.array([0,1], dtype=np.int16)
    dy = np.array([1,0], dtype=np.int16)
    for _ in range(5):
        next_point = update_point(contour, point, dx, dy)
        point = next_point
        if np.array_equal(point, next_point):
            break
        else:
            point = next_point

    return point

# Get all x,y of points in contour
def get_points_on_line(img, contour):
    points = []
    for j in range(len(contour)):
        y,x = contour[j,0,:]
        p0 = np.array([y,x], dtype = np.int16)
        pf = optimize_point(contour, p0)
        points.append(pf)
        # cv.circle(img, pf, 2, 155, 2)
    # uc.showImage(img)
    return np.array(points, dtype=np.int64)

def get_neighbor_matrix(points, max_dist=100):
    neighbors = np.zeros((len(points),2), dtype=np.int64)
    for j in range(len(points)):
        vectors = points-points[j,:]
        dists = np.sum(np.square(vectors), axis=1)
        pos_idx = dists.argsort()[1] # index of closest neighbor
        tangent = points[pos_idx, :]-points[j,:] # tangent defines positive direction

        # Find signed dists
        signed_dists = np.sign(np.inner(tangent, vectors))*dists
        neg_idx = np.argmin(1/(signed_dists+.1)) # index of closest neighbor in negative direction
        min_signed_dist = signed_dists[neg_idx]
        if min_signed_dist>=0:
            neg_idx = -1
        elif abs(min_signed_dist)>max_dist**2:
            neg_idx = min_signed_dist

        # Add to neighbors
        neighbors[j,:] = [pos_idx, neg_idx]

        # What if not unique min/max?
    return neighbors

def order_centroids_by_neighbor(points, neighbors):
    jstart, jend = np.where((neighbors<0).any(axis=1))[0] #endpoints
    j = jstart
    jorder = [jstart]
    while j!=jend:
        j1, j2 = neighbors[j, :]
        j = j1 if (j1!=-1 and (j1 not in jorder)) else j2
        jorder.append(j)
    return points[jorder,:]

def draw_points(img, points, save=False):
    img = img.copy()
    for point in points:
        cv.circle(img, point, 20, 155, -1)
    uc.showImage(img)
    if save:
        cv.imwrite('test.jpg', img)
    return


# Remove duplicates
def remove_duplicate_points(points, min_dist):
    jremove = set()
    for j in range(len(points)):
        if j not in jremove:
            vectors = points-points[j,:]
            dists = np.sum(np.square(vectors), axis=1)
            jremove |= set(np.where(dists<min_dist**2)[0])
            jremove.remove(j) #keep current point
    return np.delete(points, list(jremove), 0)

def smooth_curve(points, max_perp_distance):
    npoints = len(points)
    jkeep = []
    jcur = 0
    jtest = jcur+2
    jkeep.append(jcur)
    while jtest < npoints:
        tangent = points[jtest] - points[jcur]
        vectors = points[jcur+1:jtest] - points[jcur]
        perp_distances = [np.abs(np.cross(vector, tangent))/norm(tangent) for vector in vectors]

        if max(perp_distances)>max_perp_distance:
            jcur = jtest-1
            jtest = jcur+2
            jkeep.append(jcur)
        else:
            jtest += 1
    jkeep.append(npoints-1)
    # print(f'Smoothed curve from {npoints} to {len(jkeep)} points')
    return points[jkeep]

def build_contour_from_points(points, linewidth=2):
    npoints = len(points)
    rot90 = np.array([[0,1],[-1,0]])
    contour = np.zeros((2*npoints, 1, 2), np.int32)
    for j in range(npoints):
        if j!=(npoints-1):
            vector = points[j+1,:] - points[j,:]
            offset = rot90.dot(vector)/norm(vector)*linewidth
        contour[j,0,:] = points[j,:] + offset
        contour[-j-1,0,:] = points[j,:] - offset
    
    return contour