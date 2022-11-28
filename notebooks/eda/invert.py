import cv2

img = cv2.imread('/home/mithil/PycharmProjects/Pestedetec2.0/id_0b578a8ca2f8d5a6c3fc0f88.jpg')
img = cv2.resize(img, (3000, 3000), )
cv2.imwrite('/home/mithil/PycharmProjects/Pestedetec2.0/try.jpg', img)
img = cv2.imread('/home/mithil/PycharmProjects/Pestedetec2.0/data/images/id_0b578a8ca2f8d5a6c3fc0f88.jpg')
img = cv2.resize(img, (3000, 3000), )
cv2.imwrite('/home/mithil/PycharmProjects/Pestedetec2.0/try2.jpg', img)