def distance_transform(bin_img, metric):
    ...


B = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 0, 1, 1],
     [1, 1, 1, 1, 1]]
for row in distance_transform(B, "d4"):
    print(row)
