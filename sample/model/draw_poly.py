import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

# Generate random points
num_points = 15
points = np.random.rand(num_points, 2) * 100

# Compute the Convex Hull
hull = ConvexHull(points)

# Extract the points forming the hull
hull_points = points[hull.vertices]
hull_points = np.append(hull_points, [hull_points[0]], axis=0)  # Repeat the first point to close the hull

# Calculate the distance between each pair of points for parametrization
dist = np.sqrt(np.diff(hull_points[:, 0])**2 + np.diff(hull_points[:, 1])**2)
dist = np.insert(dist, 0, 0)  # Insert 0 distance for the first point
cumulative_dist = np.cumsum(dist)  # Cumulative distance for each point

# Generate many points for a smooth curve
num_smooth_points = 200
t_new = np.linspace(cumulative_dist[0], cumulative_dist[-1], num_smooth_points)
spl_x = interp1d(cumulative_dist, hull_points[:, 0], kind='cubic')
spl_y = interp1d(cumulative_dist, hull_points[:, 1], kind='cubic')

# Create the smooth hull points
smooth_hull_points_x = spl_x(t_new)
smooth_hull_points_y = spl_y(t_new)

# Plotting the smooth hull
fig, ax = plt.subplots()

# Plot the original points
ax.plot(points[:, 0], points[:, 1], 'o')

# Plot the smooth convex hull
ax.plot(smooth_hull_points_x, smooth_hull_points_y, 'r-', lw=2)

# Fill the smooth convex hull with a light blue color
ax.fill(smooth_hull_points_x, smooth_hull_points_y, 'skyblue', alpha=0.5)

# Set the aspect of the plot to be equal
ax.set_aspect('equal')

# Show the plot
plt.show()

import cv2

# Convert the smooth hull points to proper format for polylines and fillPoly
smooth_hull_points = np.array([np.column_stack((smooth_hull_points_x, smooth_hull_points_y))], dtype=np.int32)

# Create an empty black image
image = np.zeros((150, 150, 3), dtype=np.uint8)

# Draw the smooth convex hull on the image - we use fillPoly for a filled convex hull
cv2.fillPoly(image, smooth_hull_points, (255, 255, 255))

# Convert to grayscale to get a binary image
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the image
cv2.imshow('Smooth C'
           'onvex Hull', image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
