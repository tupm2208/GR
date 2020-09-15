import cv2

def draw_grid(image_path, grid=(32, 32)):
    grid_x, grid_y = grid
    print(image_path)
    image = cv2.imread(image_path)
    print(image)
    h, w, _ = image.shape

    interval = w/grid_x
    x_coords = [int(interval*i) for i in range(grid_x)]
    interval = h/grid_y
    y_coords = [int(interval*i) for i in range(grid_y)]

    for x in x_coords:
        cv2.line(image, (x, 0), (x,h), (0, 0, 0), 3)
    
    for y in y_coords:
        cv2.line(image, (0, y), (w,y), (0, 0, 0), 3)
    
    # cv2.imshow('img', image)
    # cv2.waitKey(0)

draw_grid('/home/tupm/Pictures/1.jpg')

