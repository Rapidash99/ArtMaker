import cv2
import numpy as np

from random import randint, random
from numba import njit


width = 512
height = 512
polygons_in_DNA = 50
polygons_max_points = 7
populations = 1000


"""
Class that represents the chromosomes
"""


class Polygons:
    polygons = []

    def __init__(self, polygons):
        self.polygons = polygons

    def replace(self, second):
        for i in range(len(second.polygons)):
            for j in range(len(second.polygons[i][0])):
                for k in range(len(second.polygons[i][0][j])):
                    self.polygons[i][0][j][k] = second.polygons[i][0][j][k]
            for j in range(len(second.polygons[i][1])):
                self.polygons[i][1][j] = second.polygons[i][1][j]


"""
Polygons initialization method
"""


def polygons_init():
    polygons = []
    for i in range(polygons_in_DNA):
        points, colour = generate_polygon()
        polygons.append([points, colour])

    object_polygons = Polygons(polygons)
    return object_polygons


"""
Polygon mutation method
"""


def mutate(polygons):
    polygon_ind = np.random.randint(0, polygons_in_DNA)
    mutation = random()
    if mutation >= 0.5:
        # mutate colour
        [r, g, b] = generate_colour()
        polygons.polygons[polygon_ind][1][0] = r
        polygons.polygons[polygon_ind][1][1] = g
        polygons.polygons[polygon_ind][1][2] = b
    else:
        # mutate point
        [x, y] = generate_point()
        point = np.random.randint(0, len(polygons.polygons[polygon_ind][0]))
        polygons.polygons[polygon_ind][0][point] = [x, y]

    return polygons


"""
Polygon crossover method
"""


def crossover(polygons):
    first = np.random.randint(0, polygons_in_DNA)
    second = np.random.randint(0, polygons_in_DNA)

    to_cross = random()
    if to_cross >= 0.5:
        # crossover colours
        colour = np.random.randint(0, 2)
        temp = polygons.polygons[first][1][colour]
        polygons.polygons[first][1][colour] = polygons.polygons[second][1][colour]
        polygons.polygons[second][1][colour] = temp
    else:
        # crossover points
        first_point = np.random.randint(0, len(polygons.polygons[first][0]))
        second_point = np.random.randint(0, len(polygons.polygons[second][0]))
        temp = polygons.polygons[first][0][first_point]
        polygons.polygons[first][0][first_point] = polygons.polygons[second][0][second_point]
        polygons.polygons[second][0][second_point] = temp

    return polygons


"""
Method that randomizing to mutate or crossover
"""


def mutate_or_crossover(polygons):
    temp = random()
    if temp >= 0.5:
        return mutate(polygons)
    else:
        return crossover(polygons)


"""
Supporting methods to generate something
"""


@njit()
def generate_colour():
    red = randint(0, 255)
    green = randint(0, 255)
    blue = randint(0, 255)

    return [red, green, blue]


@njit()
def generate_point():
    x = randint(0, width)
    y = randint(0, height)

    return [x, y]


@njit()
def generate_polygon():
    # number_of_points = randint(3, polygons_max_points)
    number_of_points = 3
    points = []
    for i in range(number_of_points):
        point = generate_point()
        points.append(point)
    colour = generate_colour()

    return points, colour


"""
Fitness calculating function. Using Numba's nopython jit, so very fast
"""


@njit()
def fitness(first, second):
    fit = 0.0
    for i in range(width):
        for k in range(height):
            f_blue = first[i][k][0]
            f_green = first[i][k][1]
            f_red = first[i][k][2]

            s_blue = second[i][k][0]
            s_green = second[i][k][1]
            s_red = second[i][k][2]

            d_blue = f_blue - s_blue
            d_green = f_green - s_green
            d_red = f_red - s_red

            pixel_dif = np.sqrt(d_blue * d_blue + d_green * d_green + d_red * d_red)
            fit += pixel_dif
    res = (fit*100)/(width*height*3*255)
    return res


"""
Method to create a picture by existing polygons
"""


def draw_pic(polygons):
    new = np.zeros((512, 512, 3), np.uint8)

    for i in range(polygons_in_DNA):
        points = np.array(polygons.polygons[i][0], np.int32)
        new = cv2.fillPoly(new, [points],
                           (polygons.polygons[i][1][0], polygons.polygons[i][1][1], polygons.polygons[i][1][2]))

    return new


"""
Method to read a path to image, open it and return
"""


def read():
    print("Please, write down the path to the image below")
    src = input()
    img = cv2.imread(src)

    if img is None:
        img = read()
    return img, src


"""
Method for testing that checks are 2 polygons the same
"""


def is_same(first, second):
    res = True
    for i in range(len(first)):
        for j in range(len(first[i])):
            for k in range(len(first[i][j])):
                if first[i][j][k] != second[i][j][k]:
                    res = False

    return res


"""
Main method to put everything together to work
"""


def main():
    img, src = read()
    parent_polygons = polygons_init()
    parent_pic = draw_pic(parent_polygons)
    parent_fit = fitness(img, parent_pic)

    number_of_evo = 0
    percents = 0

    while percents < 95 and number_of_evo < populations * 100:
        child_polygons = polygons_init()

        child_polygons.replace(parent_polygons)

        child_polygons = mutate_or_crossover(child_polygons)

        child_pic = draw_pic(child_polygons)
        child_fit = fitness(img, child_pic)

        if child_fit > parent_fit:
            parent_polygons.replace(child_polygons)
            parent_fit = child_fit
            parent_pic = child_pic

            number_of_evo += 1

            temp = int(parent_fit * 100)/100
            if temp > percents:
                percents = temp
                print(percents, '%')

        if number_of_evo % 100 == 0:
            out = 'res/' + src + ' generation' + str(number_of_evo/100 + 1) + '.png'
            cv2.imwrite(out, parent_pic)


main()
